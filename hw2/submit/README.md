# AI Capstone HW2

NYCU AI Capstone HW2

## Requirements
open3d  
numpy==1.26.4

## 1. Data Collection
Enter conda environment `habitat` and run `load.py` to collect frames of the apartment_0
```bash
conda activate habitat
python load.py
```

## 2. 3D Reconstruction
Switch to the reconstruction environment named `open3d_env` to avoid potential library conflicts
```bash
# Leave habitat
conda deactivate
conda activate open3d_env

# Reconstruct Floor 1
python reconstruct.py -f 1 -v open3d
```
