# Sliding-window detection under simulated turbulence

## Motivation
The paper "Automatic object detection in atmospheric turbulence-affected environments" (2025) studies detecting objects in imagery degraded by atmospheric turbulence. This small experiment checks, on a synthetic stand-in, whether training a detector on turbulence-degraded frames helps it keep detecting objects as turbulence gets stronger.

## Question
Can a sliding-window detector trained on turbulence-degraded frames keep a higher detection rate than the same detector trained on clean frames as simulated turbulence strength increases?

## Dataset
scikit-learn's built-in digits images (no download), split once by image into train and test (stratified, 25% test). Each digit is upscaled to 16x16 and pasted at a random position on a noisy 48x48 scene. Turbulence (levels 0-4) is a smooth random pixel-displacement warp plus Gaussian blur. This is not the paper's dataset.

## Baseline vs improved
- Baseline: sliding-window detector (HOG-style gradient features + default LightGBM object/background patch classifier) trained on clean scenes only.
- Improved: same detector, features and number of training scenes, but trained on scenes degraded with turbulence across levels 0-4, with a small LightGBM grid tuned by grouped cross-validation on the training set.

Both are evaluated on the same held-out test scenes at every level. Metric: detection rate (best-scoring window has IoU >= 0.5 with the true box), averaged over the five levels; secondary: rate at the strongest level and the per-level curve. Models are run with seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> P Hill, A Achim, D Bull (2025). *Automatic object detection in atmospheric turbulence-affected environments*. Automatic Target. https://www.spiedigitallibrary.org/proceedings/Download?urlId=10.1117%2F12.3053981

Paper author: Prof. Alin Achim, University of Bristol (alin.achim@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
