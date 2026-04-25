# AI Capstone HW2

NYCU AI CAPSTONE 2026 Spring

Spec: [Google Docx](https://drive.google.com/file/d/1aOYWcycS2J2hnm5BWxa438pOT0doBfcP/view?usp=sharing)

## 1. Preparation
The replica dataset, you can use the same one in `hw0`.

## 2. Environment Setup (Critical)
To avoid **Segmentation Faults** and library conflicts 

- **Python Version**: 3.9 or 3.10 is recommended.
- **NumPy Version**: You **MUST** use `numpy==1.26.4`. (Do NOT use 2.0+).

## Phase 1: Data Collection
Enter conda environment `habitat` and run `load.py` to collect RGB-D images and ground truth poses of the apartment_0.
```bash
conda activate habitat
python load.py
```

## Phase 2: 3D Reconstruction
Switch to the reconstruction environment before running the following commands.  
⚠️ Requirements:  
open3d  
numpy==1.26.4

### Standard Version (Open3D ICP)
Use Open3D's built-in ICP algorithm for reconstruction.
```bash
# Reconstruct Floor 1
python reconstruct.py -f 1 -v open3d

# Reconstruct Floor 2
python reconstruct.py -f 2 -v open3d
```
### Bonus Version (Custom ICP Implementation)
If implementing the own ICP algorithm, use the my_icp option.
```bash
# Run reconstruction with your own ICP
python reconstruct.py -f 1 -v my_icp
```
