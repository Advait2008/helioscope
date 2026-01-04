import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import sys
import os

# === CONFIG ===
MODEL_PATH = ""  # please paste saved model path here
PATCH_SIZE = 128
STRIDE = 128
PIXEL_AREA_M2 = 0.09

# === MATCH TRAINING NORMALIZATION ===
transform = A.Compose([
    A.Resize(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# === LOAD MODEL ===
def load_model():
    model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# === PATCHIFY FUNCTION (LIKE TRAINING) ===
def extract_patches(image, size=PATCH_SIZE, stride=STRIDE):
    H, W, _ = image.shape
    patches, positions = [], []
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            patch = image[y:y+size, x:x+size]
            patches.append(patch)
            positions.append((y, x))
    return patches, positions, H, W

# === RUN MODEL ON PATCHES ===
def infer_full_mask(model, image):
    patches, positions, H, W = extract_patches(image)
    mask_out = np.zeros((H, W), dtype=np.uint8)
    total_rooftop_pixels = 0

    for patch, (y, x) in zip(patches, positions):
        transformed = transform(image=patch)
        input_tensor = transformed["image"].unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor)
            pred_mask = torch.argmax(pred.squeeze(), dim=0).cpu().numpy()
            mask_out[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_mask
            total_rooftop_pixels += np.sum(pred_mask == 1)

    return mask_out, total_rooftop_pixels

# === MAIN EXECUTION ===
def main(image_path):
    print("Loading model...")
    model = load_model()

    print("Reading image...")
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3]).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))  # CHW to HWC

    print("Running inference...")
    mask, rooftop_pixels = infer_full_mask(model, image)
    rooftop_area = rooftop_pixels * PIXEL_AREA_M2

    print("Rooftop Pixels:", rooftop_pixels)
    print("Estimated Rooftop Area:", round(rooftop_area, 2), "m²")

    # Save results
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Predicted Rooftops\nArea: {rooftop_area:.2f} m²")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("results.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Return results as JSON-like output for Streamlit to parse
    print(f"RESULTS_START")
    print(f"ROOFTOP_PIXELS:{rooftop_pixels}")
    print(f"ROOFTOP_AREA:{rooftop_area:.2f}")
    print(f"RESULTS_END")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python modeltester_UI.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    
    main(image_path) 