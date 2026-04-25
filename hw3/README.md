# AI Capstone HW3

NYCU AI CAPSTONE 2026 Spring

Spec: [Google Docx](https://docs.google.com/document/d/1-BvmwVXlk8g06hZYU2sgTaR3iAti8XRA73k6sIdHrrA/edit?usp=sharing)

## 1. Preparation
The replica dataset, you can use the same one in `hw0`.

## 2. Environment Setup
Run the code in conda environment `habitat`

## Basic Version: Original RRT
Enter conda environment `habitat`. After setting the parameter `adaptive` to `False` in `main.py`, run `main.py`
```bash
conda activate habitat
python main.py
```

Choose one point on the image shown on the screen to be the start point, then the program will execute path planning and use this path to do robot navigation.

## Advanced Version: Adaptive goal-bias RRT
Enter conda environment `habitat`. After setting the parameter `adaptive` to `True` in `main.py`, run `main.py`
```bash
conda activate habitat
python main.py
```

The procedure afterwards will be the same as original RRT.