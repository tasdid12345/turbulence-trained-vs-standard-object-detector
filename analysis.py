"""Sliding-window detection under simulated turbulence: clean-trained vs turbulence-trained."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import zoom, gaussian_filter, map_coordinates
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from lightgbm import LGBMClassifier

SEEDS, LEVELS, S, B, CELL = [42, 43, 44], 5, 48, 16, 4  # scene size, box size, HOG cell
NW = (S - B) // CELL + 1  # windows per axis (stride = CELL)
np.random.seed(42)

def make_scene(digit, rng):
    """Noisy background with an upscaled digit pasted at a random position."""
    img = np.clip(0.3 + 0.12 * rng.standard_normal((S, S)), 0, 1)
    r, c = rng.integers(0, S - B + 1, 2)
    img[r:r + B, c:c + B] = np.clip(img[r:r + B, c:c + B] + 0.7 * zoom(digit / 16.0, B / 8, order=1), 0, 1)
    return img, (r, c)

def turb(img, level, rng):
    """Smooth random pixel-displacement warp + Gaussian blur, both scaled by level."""
    if level == 0:
        return img
    d = [gaussian_filter(rng.standard_normal(img.shape), 5) for _ in range(2)]
    d = [0.6 * level * f / f.std() for f in d]
    yy, xx = np.mgrid[:S, :S].astype(float)
    out = map_coordinates(img, [yy + d[0], xx + d[1]], order=1, mode="reflect")
    return gaussian_filter(out, 0.5 * level)

def hog_windows(scenes):
    """HOG-style features (9 unsigned orientation bins, 4x4 cells) for every window: (N, NW*NW, 144)."""
    n = len(scenes)
    gy, gx = np.gradient(scenes.astype(np.float32), axis=(1, 2))
    mag = np.hypot(gx, gy)
    b = np.minimum((np.arctan2(gy, gx) % np.pi) / np.pi * 9, 8).astype(int)
    hist = mag[..., None] * (b[..., None] == np.arange(9))
    cells = hist.reshape(n, S // CELL, CELL, S // CELL, CELL, 9).sum((2, 4))
    win = sliding_window_view(cells, (B // CELL, B // CELL), axis=(1, 2)).reshape(n, NW * NW, -1)
    return win / np.sqrt((win ** 2).sum(-1, keepdims=True) + 0.25)  # soft window L2 normalisation

def window_iou(pos):
    """IoU of every sliding window with the true box: (N, NW*NW)."""
    wr, wc = [a.ravel() * CELL for a in np.meshgrid(np.arange(NW), np.arange(NW), indexing="ij")]
    pos = np.asarray(pos)
    inter = np.clip(B - abs(wr - pos[:, :1]), 0, None) * np.clip(B - abs(wc - pos[:, 1:]), 0, None)
    return inter / (2 * B * B - inter)

def train_set(feats, iou):
    """Patch labels: positive IoU>=0.6, negative IoU<0.4, ambiguous windows dropped."""
    keep = ((iou >= 0.6) | (iou < 0.4)).ravel()
    groups = np.repeat(np.arange(len(iou)), iou.shape[1])[keep]
    return feats.reshape(-1, feats.shape[-1])[keep], (iou.ravel() >= 0.6)[keep].astype(int), groups

def detection_rate(model, feats, iou):
    """Fraction of scenes whose top-scoring window has IoU >= 0.5."""
    p = model.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1].reshape(len(feats), -1)
    return float((iou[np.arange(len(iou)), p.argmax(1)] >= 0.5).mean())

# ---- data: split by image once, then build scenes ----
digits = load_digits()
tr_idx, te_idx = train_test_split(np.arange(len(digits.target)), test_size=0.25, stratify=digits.target, random_state=42)
rng = np.random.default_rng(42)
tr = [make_scene(digits.images[i], rng) for i in tr_idx]
te = [make_scene(digits.images[i], rng) for i in te_idx]
iou_tr, iou_te = window_iou([p for _, p in tr]), window_iou([p for _, p in te])
print(f"train scenes {len(tr)}, test scenes {len(te)}", flush=True)

# training scenes: baseline = clean; improved = same scenes, level cycled over 0..4 (same count)
clean_tr = np.stack([s for s, _ in tr])
turb_tr = np.stack([turb(s, i % LEVELS, np.random.default_rng([43, i])) for i, (s, _) in enumerate(tr)])
X_b, y_b, g_b = train_set(hog_windows(clean_tr), iou_tr)
X_i, y_i, g_i = train_set(hog_windows(turb_tr), iou_tr)
# test scenes: identical degraded test set for both models at every level
test_feats = [hog_windows(np.stack([turb(s, L, np.random.default_rng([44, L, i])) for i, (s, _) in enumerate(te)]))
              for L in range(LEVELS)]
print("features built", flush=True)

base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
default = dict(num_leaves=31, n_estimators=100, min_child_samples=20)  # standard LightGBM defaults
grid = {"num_leaves": [7, 15, 31], "n_estimators": [100, 200], "min_child_samples": [20, 50]}
gs = GridSearchCV(LGBMClassifier(random_state=42, **base), grid, cv=GroupKFold(3), scoring="average_precision")
gs.fit(X_i, y_i, groups=g_i)  # CV on training data only (grouped by scene)
tuned = gs.best_params_
print("CV-tuned params:", tuned, flush=True)

res = {"baseline": [], "improved": []}
for seed in SEEDS:
    for name, (X, y), params in [("baseline", (X_b, y_b), default), ("improved", (X_i, y_i), tuned)]:
        m = LGBMClassifier(random_state=seed, **base, **params).fit(X, y)
        res[name].append([detection_rate(m, test_feats[L], iou_te) for L in range(LEVELS)])
    print(f"seed {seed} done", flush=True)

R = {k: np.array(v) for k, v in res.items()}  # (seeds, levels)
main = {k: R[k].mean(1) for k in R}  # per-seed mean over levels
metrics = {}
for k in R:
    metrics[f"{k}_score"] = float(main[k].mean())
    metrics[f"{k}_score_std"] = float(main[k].std())
    metrics[f"{k}_rate_strongest_level"] = float(R[k][:, -1].mean())
    metrics[f"{k}_rate_strongest_level_std"] = float(R[k][:, -1].std())
    for L in range(LEVELS):
        metrics[f"{k}_rate_level{L}"] = float(R[k][:, L].mean())
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
metrics["improvement_strongest_level"] = metrics["improved_rate_strongest_level"] - metrics["baseline_rate_strongest_level"]
metrics["chance_rate_random_window"] = float((iou_te >= 0.5).mean())
if not all(np.isfinite(v) for v in metrics.values()) or metrics["baseline_rate_level0"] <= metrics["chance_rate_random_window"]:
    sys.exit("metrics not meaningful")

os.makedirs("figures", exist_ok=True)
for k in R:
    plt.errorbar(range(LEVELS), R[k].mean(0), R[k].std(0), marker="o", label=k)
plt.axhline(metrics["chance_rate_random_window"], ls=":", c="gray", label="random window")
plt.xlabel("turbulence level"); plt.ylabel("detection rate (IoU>=0.5)"); plt.legend()
plt.savefig("figures/detection_vs_turbulence.png", dpi=120, bbox_inches="tight")

out = {
    "question": "Can a sliding-window detector trained on turbulence-degraded frames keep a higher detection rate than the same detector trained on clean frames as simulated turbulence strength increases?",
    "method": "Digits pasted on noisy 48x48 scenes; turbulence = smooth random warp + Gaussian blur at levels 0-4; HOG-style features + LightGBM patch classifier in a sliding window (stride 4); detection = top-scoring window has IoU>=0.5. Baseline trained on clean scenes, improved on turbulence-degraded scenes with CV-tuned classifier. Mean of seeds 42-44.",
    "dataset": {"name": "sklearn digits (synthetic surveillance scenes)", "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                "n_rows": int(len(digits.target)), "n_train": len(tr), "n_test": len(te), "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "detection rate (IoU>=0.5), mean over turbulence levels 0-4",
    "higher_is_better": True,
    "baseline": "Sliding-window detector (HOG-style features + default LightGBM patch classifier) trained on clean scenes only.",
    "improved": "Same detector and number of training scenes, but scenes are degraded with turbulence levels 0-4 and LightGBM hyper-parameters are tuned by grouped 3-fold CV on the training set.",
    "metrics": metrics,
    "notes": f"Synthetic scenes and simulated turbulence; results may not transfer to real imagery. Improved model differs from baseline in both training-data degradation and CV tuning (tuned params: {tuned}), so gains are not attributable to augmentation alone. Test scenes are fixed across seeds; std reflects LightGBM seed only. Stride-4 windows on 16x16 boxes. Per-level metrics are seed means.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(metrics, indent=2))
