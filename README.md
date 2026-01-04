# Helioscope

Helioscope is a Python-based system that estimates rooftop solar potential in dense urban environments using satellite imagery. It is designed as a planning and decision-support tool, not a precision engineering model.

The goal of Helioscope is to reduce uncertainty around solar adoption by translating technical analysis into human-readable outputs such as energy generation, carbon reduction, and return-on-investment timelines.

---

## Process flow
- Accepts satellite imagery of urban areas
- Identifies viable rooftop regions using a segmentation model
- Estimates:
  - Potential energy generation
  - Carbon emission reduction
  - Approximate financial break-even time
  - 
## Notes & limitations
- Irradiance and location data are currently hardcoded for select U.S. locations. Future iterations will integrate live APIs to support continuous, location-specific data (including India)
- While I am comfortable building and iterating in code, this is my first time formally organizing a project on GitHub. I appreciate your patience if the repository structure reflects a learning curve rather than a lack of rigor.

---

## How to run
   - pip install -r requirements.txt 
   - streamlit run helioscopefinal.py
