import streamlit as st
import subprocess
import os
import tempfile
import re
from PIL import Image
import io
import rasterio
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Helioscope - Rooftop Detection",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .slogan {
        font-size: 1.2rem;
        color: #666;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .results-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">Helioscope</div>
    <div class="slogan">From pixels to panels. Terraces to terawatts.</div>
</div>
""", unsafe_allow_html=True)

# === LOCATION IRRADIANCE DATA ===
LOCATION_IRRADIANCE = {
    "Washington, D.C.": 4.7,   # kWh/m²/day
    "Houston, TX": 5.27,
    "Dallas, TX": 5.47,
    "Austin, TX": 5.41
}

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📁 Upload Satellite Image")
    st.markdown("Upload a `.tif` satellite image to detect rooftops and calculate potential solar panel areas.")
    
    # Location dropdown moved to main app
    location = st.selectbox(
        "Select Location",
        ["Washington, D.C.", "Houston, TX", "Dallas, TX", "Austin, TX"],
        help="Choose the location for irradiance calculations"
    )
    
    uploaded_file = st.file_uploader(
        "Choose a TIF file",
        type=['tif', 'tiff'],
        help="Upload a satellite image in TIF format"
    )

with col2:
    st.markdown("### 🎯 Detection Settings")
    st.markdown(f"**Selected Location:** {location}")
    st.markdown("**Model:** DeepLabV3+ ResNet34")
    st.markdown("**Resolution:** High Precision")

# Display uploaded image if available
if uploaded_file is not None:
    st.markdown("### 📸 Uploaded Image Preview")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_image_path = tmp_file.name
    
    # Display the image
    try:
        with rasterio.open(temp_image_path) as src:
            # Read RGB bands
            image_data = src.read([1, 2, 3])
            # Normalize for display
            image_display = np.transpose(image_data, (1, 2, 0))
            image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())
            image_display = (image_display * 255).astype(np.uint8)
            
            st.image(image_display, caption="Uploaded Satellite Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")

# Run detection button
if uploaded_file is not None:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Run Rooftop Detection", use_container_width=True)
    
    if run_button:
        with st.spinner("🔍 Analyzing satellite image..."):
            try:
                # Run the model script
                result = subprocess.run(
                    ["python", "modeltester_UI.py", temp_image_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    # Parse the output
                    output_lines = result.stdout.split('\n')
                    
                    # Extract results
                    rooftop_pixels = None
                    rooftop_area = None
                    
                    for line in output_lines:
                        if line.startswith("ROOFTOP_PIXELS:"):
                            rooftop_pixels = int(line.split(":")[1])
                        elif line.startswith("ROOFTOP_AREA:"):
                            rooftop_area = float(line.split(":")[1])
                    
                    if rooftop_pixels is not None and rooftop_area is not None:
                        st.success("✅ Detection completed successfully!")
                        
                        # Display results
                        st.markdown("### 📊 Detection Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">{:,}</div>
                                <div class="metric-label">Rooftop Pixels</div>
                            </div>
                            """.format(rooftop_pixels), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">{:.1f}</div>
                                <div class="metric-label">Rooftop Area (m²)</div>
                            </div>
                            """.format(rooftop_area), unsafe_allow_html=True)
                        
                        # Display the results image if it exists
                        if os.path.exists("results.png"):
                            st.markdown("### 🖼️ Detection Visualization")
                            st.image("results.png", caption="Original Image vs Predicted Rooftops", use_column_width=True)
                        
                        # Additional insights
                        st.markdown("### 💡 Solar Potential Insights")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Potential Panels", f"{int(rooftop_area / 20)}", "~20m² per panel")
                        
                        with col2:
                            # Use irradiance for selected location
                            irradiance = LOCATION_IRRADIANCE.get(location, 5.0)  # fallback if not found
                            # 1 m² * irradiance (kWh/m²/day) * 365 days * 20% panel efficiency = annual kWh
                            # For power, keep as before (0.15 kW per m²)
                            potential_power = rooftop_area * 0.15  # 150W per m²
                            st.metric("Potential Power", f"{potential_power:.0f} kW", f"{irradiance} kWh/m²/day")
                        
                        with col3:
                            # Annual energy = rooftop_area * irradiance * 365 * 0.20 (20% efficiency)
                            annual_energy = rooftop_area * irradiance * 365 * 0.20 / 1000  # in MWh
                            st.metric("Annual Energy", f"{annual_energy:.1f} MWh", f"{irradiance} kWh/m²/day")
                        
                        with col4:
                            # Carbon reduction: 1 kW installed = 1.4 metric tons CO2/year
                            carbon_reduction = potential_power * 1.4
                            st.metric("CO₂ Reduction", f"{carbon_reduction:.1f} t/year", "1.4 t/kW/year")
                        
                        # === ROI & PRICING CALCULATOR SECTION ===
                        st.markdown("### 💰 Pricing & ROI Calculator")
                        # City-specific rates and install costs
                        CITY_PRICING = {
                            "Washington, D.C.": {"rate": 0.13, "install": 1.30},
                            "Houston, TX": {"rate": 0.11, "install": 1.10},
                            "Austin, TX": {"rate": 0.12, "install": 1.00},
                            "Dallas, TX": {"rate": 0.12, "install": 1.05},
                        }
                        # Use 20% efficiency and 85% loss factor (typical for solar)
                        panel_efficiency = 0.20
                        loss_factor = 0.85
                        city_data = CITY_PRICING.get(location, {"rate": 0.12, "install": 1.10})
                        electricity_rate = city_data["rate"]
                        install_cost_per_watt = city_data["install"]
                        # Calculations
                        capacity_kw = rooftop_area * panel_efficiency  # kW
                        annual_output_kwh = capacity_kw * irradiance * 365 * loss_factor
                        annual_savings = annual_output_kwh * electricity_rate
                        install_cost = capacity_kw * install_cost_per_watt * 1000  # kW to W
                        roi = annual_savings / install_cost if install_cost > 0 else 0
                        payback_years = install_cost / annual_savings if annual_savings > 0 else 0
                        # Display
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Install Cost", f"${install_cost:,.0f}", f"@ ${install_cost_per_watt:.2f}/W")
                        with col2:
                            st.metric("Annual Savings", f"${annual_savings:,.0f}", f"@ ${electricity_rate:.2f}/kWh")
                        with col3:
                            st.metric("ROI", f"{roi*100:.1f}%", "Annual Return")
                        with col4:
                            st.metric("Payback Period", f"{payback_years:.1f} yrs", "Years to Payback")
                        
                    else:
                        st.error("❌ Could not parse model results")
                        
                else:
                    st.error(f"❌ Model execution failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                st.error("❌ Model execution timed out. Please try with a smaller image.")
            except Exception as e:
                st.error(f"❌ Error running detection: {str(e)}")
            finally:
                # Clean up temporary file
                if 'temp_image_path' in locals():
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Helioscope - Advanced Rooftop Detection for Solar Energy Planning</p>
    <p>Powered by Deep Learning & Satellite Imagery</p>
</div>
""", unsafe_allow_html=True) 