"""Builds the two Colab notebooks from cell sources (run: python scripts/build_notebooks.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILS = (ROOT / "app" / "gesture_utils.py").read_text()


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


M, C = "markdown", "code"

# =====================================================================
# Notebook 1: generate synthetic clips
# =====================================================================
gen = [
(M, """
# A2 · Step 1: generate synthetic gesture clips (text-to-video)

**SOTA idea:** Step-Video-T2V (Stepfun, 30B parameters, Video-VAE with 16×16 spatial and 8× temporal compression).
We use it to generate short hand-gesture clips with varied backgrounds, lighting and camera motion,
cut them into frames, and use the frames (a) to augment training and (b) as a stress-test set.

The full Step-Video-T2V needs an 80 GB GPU, so there are three ways to get clips:

| Option | What runs | Cost / hardware |
|---|---|---|
| **A. `wan`** | Open T2V model **Wan2.1-T2V-1.3B** in this notebook (lighter alternative; Step-Video idea credited) | Colab **L4/A100** or **Kaggle** (29 GB RAM). Free Colab T4 may run out of RAM. |
| **B. `replicate`** | Real **Step-Video-T2V** through the Replicate API | Paid per run, needs a Replicate token |
| **C. `manual`** | Clips made in any online Step-Video / T2V demo, uploaded to Drive | Free |

All options end the same way: `.mp4` files in `CLIPS_DIR/<class>/`. The main notebook starts from there.
⚠️ T2V models often draw hands wrongly. The main notebook has a manual check step: keep only clips where the gesture is correct.
"""),
(C, """
GENERATOR = "wan"          # "wan" | "replicate" | "manual"
CLIPS_PER_CLASS = 12       # 12 x 3 classes = 36 clips
RUN_ON = "colab"           # "colab" (saves to Drive) | "kaggle" (saves to /kaggle/working, download after)

import os
if RUN_ON == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    CLIPS_DIR = "/content/drive/MyDrive/comp-vis1/a2/synthetic_clips"
else:
    CLIPS_DIR = "/kaggle/working/synthetic_clips"
os.makedirs(CLIPS_DIR, exist_ok=True)
print("Clips will be saved to", CLIPS_DIR)
"""),
(M, "## Prompts\nEach prompt = precise gesture description × environment × camera motion. The environment tag is saved so we can report accuracy per environment."),
(C, """
import itertools, random, csv

GESTURES = {
    "peace-sign": "a human hand making a peace sign, index and middle fingers raised in a V shape, other fingers folded",
    "thumbs-up":  "a human hand giving a thumbs up, closed fist with the thumb pointing straight up",
    "stop":       "a human hand showing a stop gesture, open palm facing the camera, all five fingers extended and held together",
}
ENVIRONMENTS = {
    "dark_room":   "in a dark room lit only by a dim phone screen, low exposure",
    "sunny_street":"outdoors on a sunny city street, harsh sunlight",
    "kitchen":     "in a cluttered home kitchen, warm indoor light",
    "office":      "in an office with cold fluorescent light",
    "night_neon":  "outdoors at night under colorful neon signs",
    "plain_wall":  "in front of a plain white wall, soft light",
    "classroom":   "in a university classroom with desks in the background",
    "car":         "inside a car, daylight through the windows",
}
CAMERA = {
    "static":   "static camera",
    "pan":      "slow camera pan from left to right",
    "handheld": "handheld shaky camera, slight motion blur",
    "zoom":     "slow zoom in towards the hand",
}
STYLE = "realistic smartphone video, close-up, the hand is centered and fully visible, natural skin, 4 seconds"
NEGATIVE = ("extra fingers, missing fingers, deformed hand, fused fingers, blurry hand, cartoon, "
            "text, watermark, low quality, distorted anatomy")

random.seed(42)
jobs = []
combos = list(itertools.product(ENVIRONMENTS, CAMERA))
for cls, g in GESTURES.items():
    for i, (env, cam) in enumerate(random.sample(combos, CLIPS_PER_CLASS)):
        prompt = f"{g}, {ENVIRONMENTS[env]}, {CAMERA[cam]}, {STYLE}"
        jobs.append(dict(cls=cls, env=env, camera=cam, prompt=prompt,
                         file=f"{CLIPS_DIR}/{cls}/{cls}__{env}__{cam}__{i:02d}.mp4"))

with open(f"{CLIPS_DIR}/prompts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=jobs[0].keys()); w.writeheader(); w.writerows(jobs)
print(len(jobs), "jobs. Example:\\n", jobs[0]["prompt"])
"""),
(M, "## Option A: Wan2.1-T2V-1.3B (open model, runs here)\nAlready finished clips are skipped, so if the runtime disconnects you can re-run the cell and it continues."),
(C, """
if GENERATOR == "wan":
    !pip install -q -U diffusers transformers accelerate ftfy imageio imageio-ffmpeg
"""),
(C, """
if GENERATOR == "wan":
    import torch, time
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()      # fits a 16 GB GPU
    print("GPU:", torch.cuda.get_device_name(0), "| dtype:", dtype)

    for j in jobs:
        if os.path.exists(j["file"]):
            continue
        os.makedirs(os.path.dirname(j["file"]), exist_ok=True)
        t0 = time.time()
        frames = pipe(prompt=j["prompt"], negative_prompt=NEGATIVE,
                      height=480, width=832, num_frames=33,       # ~2 s at 16 fps
                      num_inference_steps=30, guidance_scale=5.0,
                      generator=torch.Generator("cuda").manual_seed(abs(hash(j["file"])) % 2**31)
                      ).frames[0]
        export_to_video(frames, j["file"], fps=16)
        print(f"{os.path.basename(j['file'])}  {time.time()-t0:.0f}s")
"""),
(M, "## Option B: real Step-Video-T2V via Replicate\nOpen https://replicate.com/zsxkib/step-video-t2v → *API* tab, copy the full version id (`zsxkib/step-video-t2v:<hash>`) and check the input names there. Each run is billed by Replicate."),
(C, """
if GENERATOR == "replicate":
    !pip install -q replicate
    import os, replicate, urllib.request
    os.environ["REPLICATE_API_TOKEN"] = "r8_..."                  # your token
    STEPVIDEO_VERSION = "zsxkib/step-video-t2v:PASTE_VERSION_HASH"  # from the API tab

    for j in jobs:
        if os.path.exists(j["file"]):
            continue
        os.makedirs(os.path.dirname(j["file"]), exist_ok=True)
        out = replicate.run(STEPVIDEO_VERSION, input={"prompt": j["prompt"],
                                                      "negative_prompt": NEGATIVE})
        url = out[0] if isinstance(out, list) else out
        url = getattr(url, "url", url)
        urllib.request.urlretrieve(str(url), j["file"])
        print("saved", j["file"])
"""),
(M, "## Option C: manual\nGenerate clips in any online T2V demo using the prompts from `prompts.csv`, and upload them to `CLIPS_DIR/<class>/` with the file names from the `file` column (the environment tag is read from the name)."),
(C, """
from pathlib import Path
for cls in GESTURES:
    n = len(list(Path(CLIPS_DIR, cls).glob("*.mp4")))
    print(f"{cls}: {n} clips")
"""),
]

# =====================================================================
# Notebook 2: main pipeline
# =====================================================================
main = [
(M, """
# Computer Vision · Assignment 2: Hand Gesture Recognition + SOTA synthetic motion
**Team:** Aruzhan Rysbayeva, Alzira Kuanyshbekova, Saltanat Sergeyeva (SE-2403)

Pipeline:
1. A1 dataset with the **same split** as A1 (fair comparison)
2. Synthetic frames from text-to-video clips (Step-Video-T2V idea; see `A2_generate_clips.ipynb`), manual check, split by clip
3. Models: A1 baseline CNN · MobileNetV2 (real only, = A1) · **MobileNetV2 + synthetic (A2)** · ablation: + synthetic without augmentation
4. Evaluation: test metrics, accuracy by **environment** (simulated conditions, held-out synthetic clips, optional new real photos)
5. Grad-CAM explanations · latency · export for the demo app

Runtime → Change runtime type → **GPU**.
"""),
(C, """
from google.colab import drive
drive.mount('/content/drive')
!pip install -q pillow-heif opencv-python-headless
"""),
(C, """
import os, json, time, random, shutil, platform
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
register_heif_opener()
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

ROOT      = Path("/content/drive/MyDrive/comp-vis1")
REAL_DIR  = ROOT / "hand_gestures"             # A1 dataset
A2        = ROOT / "a2"
CLIPS_DIR = A2 / "synthetic_clips"             # from A2_generate_clips.ipynb
SYN_DIR   = A2 / "synthetic_frames"            # extracted frames (created below)
ENV_DIR   = A2 / "test_environments"           # OPTIONAL new real photos: <env>/<class>/*.jpg
OUT       = A2 / "outputs"; FIG = OUT / "figures"
for d in (SYN_DIR, FIG): d.mkdir(parents=True, exist_ok=True)

SEED = 42
EXT = {".jpg", ".jpeg", ".png", ".heic"}
print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
print("Real data exists:", REAL_DIR.exists(), "| clips exist:", CLIPS_DIR.exists())
"""),
(M, "### Shared code (identical to `app/gesture_utils.py`)"),
(C, UTILS),
(M, "## 1. A1 dataset, same split as A1"),
(C, """
image_paths, labels = [], []
for label, c in enumerate(CLASSES):
    for f in sorted((REAL_DIR / c).iterdir()):
        if f.is_file() and f.suffix.lower() in EXT:
            image_paths.append(str(f)); labels.append(label)

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))

def load_images(paths):
    out = []
    for p in paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        out.append(to_model_input(img))
    return np.stack(out).astype(np.float32)

Xtr, Xva, Xte = load_images(X_train), load_images(X_val), load_images(X_test)
ytr, yva, yte = map(np.array, (y_train, y_val, y_test))
print(Xtr.shape, Xva.shape, Xte.shape)
"""),
(M, """
## 2. Synthetic frames from generated clips
* From each clip we take **8 evenly spaced frames**, resized to 224×224.
* **Split by clip** (never by frame): 75% of clips → training, 25% → *synthetic stress-test set*.
* Frames used for training are **not watermarked** (a watermark would be a shortcut feature). They are labelled by folder/file name and in the report; every synthetic image shown to people is watermarked.
"""),
(C, """
FRAMES_PER_CLIP = 8
rows = []
for c in CLASSES:
    for clip in sorted((CLIPS_DIR / c).glob("*.mp4")):
        parts = clip.stem.split("__")                       # cls__env__camera__idx
        env, cam = (parts[1], parts[2]) if len(parts) >= 4 else ("unknown", "unknown")
        frames = read_video_frames(clip, n=FRAMES_PER_CLIP)
        for k, fr in enumerate(frames):
            dst = SYN_DIR / c / f"{clip.stem}__f{k}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                Image.fromarray(fr).resize((224, 224)).save(dst, quality=95)
            rows.append(dict(path=str(dst), cls=c, label=CLASSES.index(c),
                             clip=clip.stem, env=env, camera=cam))
syn = pd.DataFrame(rows)
print(syn.groupby("cls").agg(clips=("clip", "nunique"), frames=("path", "count")))
"""),
(M, "### 2.1 Manual check. ⚠️ Required step\nLook at the contact sheet. Write the names of clips where the gesture is wrong (wrong fingers, deformed hand, no hand) into `REJECT_CLIPS` and run the next cell."),
(C, """
def contact_sheet(df, cls, n_cols=6):
    clips = df[df.cls == cls]["clip"].unique()
    fig, axes = plt.subplots(int(np.ceil(len(clips) / n_cols)), n_cols,
                             figsize=(n_cols * 2.6, np.ceil(len(clips) / n_cols) * 2.9), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    for ax, clip in zip(axes.flat, clips):
        p = df[df["clip"] == clip].path.iloc[FRAMES_PER_CLIP // 2]
        ax.imshow(Image.open(p)); ax.set_title(clip.split("__", 1)[1][:28], fontsize=7)
    fig.suptitle(f"{cls}: SYNTHETIC frames (middle frame of each clip)"); plt.tight_layout(); plt.show()

for c in CLASSES:
    contact_sheet(syn, c)
"""),
(C, """
REJECT_CLIPS = [
    # "thumbs-up__dark_room__pan__03",
]
syn["kept"] = ~syn["clip"].isin(REJECT_CLIPS)
print("Rejected clips:", len(REJECT_CLIPS), "| kept frames:", syn.kept.sum(), "of", len(syn))
syn = syn[syn.kept].reset_index(drop=True)

# split by clip, stratified by class
clips = syn.drop_duplicates("clip")[["clip", "label"]]
tr_clips, st_clips = train_test_split(clips["clip"], test_size=0.25, random_state=SEED,
                                      stratify=clips.label)
syn["split"] = np.where(syn["clip"].isin(tr_clips), "train", "stress")
syn.to_csv(OUT / "synthetic_frames.csv", index=False)

Xsyn = load_images(syn[syn.split == "train"].path); ysyn = syn[syn.split == "train"].label.values
Xstr = load_images(syn[syn.split == "stress"].path); ystr = syn[syn.split == "stress"].label.values
print("Synthetic train frames:", len(Xsyn), "| synthetic stress frames:", len(Xstr))
"""),
(C, """
# Dataset summary table for the report
summary = pd.DataFrame({
    "Real train": pd.Series(ytr).value_counts().sort_index().values,
    "Real val":   pd.Series(yva).value_counts().sort_index().values,
    "Real test":  pd.Series(yte).value_counts().sort_index().values,
    "Synthetic train": [int((ysyn == i).sum()) for i in range(3)],
    "Synthetic stress": [int((ystr == i).sum()) for i in range(3)],
}, index=CLASSES)
summary.loc["Total"] = summary.sum()
syn_share = len(Xsyn) / (len(Xsyn) + len(Xtr))
print(summary.to_markdown())
print(f"\\nSynthetic share of the training set: {syn_share:.1%}")
"""),
(M, """
## 3. Training
Same recipe as A1 for every MobileNetV2 variant: stage 1 (frozen base, 10 epochs, Adam) → stage 2 (last 30 layers unfrozen, Adam 1e-5, up to 15 epochs, early stopping on val loss, patience 4).
Validation is always the **real** validation set.
"""),
(C, """
def train_mobilenet(Xa, ya, use_aug=True, name="model"):
    tf.keras.utils.set_random_seed(SEED)
    m = build_mobilenet(use_aug=use_aug)
    m.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    h1 = m.fit(Xa, ya, validation_data=(Xva, yva), epochs=10, batch_size=16, verbose=2)
    base = m.get_layer("mobilenetv2"); base.trainable = True
    for l in base.layers[:-30]: l.trainable = False
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    h2 = m.fit(Xa, ya, validation_data=(Xva, yva), epochs=15, batch_size=16, callbacks=[es], verbose=2)
    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    hist["stage1_epochs"] = len(h1.history["loss"])
    m.save(OUT / f"{name}.keras")
    return m, hist

def train_baseline():
    tf.keras.utils.set_random_seed(SEED)
    m = build_baseline_cnn()
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=15, batch_size=16, verbose=2)
    return m, dict(h.history)

Xmix = np.concatenate([Xtr, Xsyn]); ymix = np.concatenate([ytr, ysyn])
models_, hists = {}, {}
models_["Baseline CNN (A1)"], hists["Baseline CNN (A1)"] = train_baseline()
models_["MobileNetV2 real (A1)"], hists["MobileNetV2 real (A1)"] = train_mobilenet(Xtr, ytr, True, "mnv2_real")
models_["MobileNetV2 + synthetic (A2)"], hists["MobileNetV2 + synthetic (A2)"] = train_mobilenet(Xmix, ymix, True, "mnv2_syn")
# Ablation: same data, augmentation switched off
models_["Ablation: + synthetic, no aug"], hists["Ablation: + synthetic, no aug"] = train_mobilenet(Xmix, ymix, False, "mnv2_syn_noaug")
"""),
(C, """
# Training curves
fig, axes = plt.subplots(2, len(hists), figsize=(4.2 * len(hists), 7))
for j, (name, h) in enumerate(hists.items()):
    for i, (k, t) in enumerate([("accuracy", "Accuracy"), ("loss", "Loss")]):
        ax = axes[i, j]
        ax.plot(h[k], label="train"); ax.plot(h["val_" + k], label="val")
        if "stage1_epochs" in h: ax.axvline(h["stage1_epochs"] - 0.5, ls="--", c="gray", lw=1)
        ax.set_title(f"{name}\\n{t}", fontsize=9); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=150); plt.show()
"""),
(M, "## 4. Evaluation on the real test set (97 images)"),
(C, """
def predict(m, X, bs=32):
    return m.predict(X, batch_size=bs, verbose=0).argmax(1)

def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Precision=pr, Recall=rc, **{"Macro-F1": f1})

preds = {n: predict(m, Xte) for n, m in models_.items()}
main_tab = pd.DataFrame({n: metrics(yte, p) for n, p in preds.items()}).T.round(3)
print(main_tab.to_markdown())

fig, axes = plt.subplots(1, len(preds), figsize=(4.3 * len(preds), 3.8))
for ax, (n, p) in zip(axes, preds.items()):
    sns.heatmap(confusion_matrix(yte, p), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(n, fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=150); plt.show()

for n in ["MobileNetV2 real (A1)", "MobileNetV2 + synthetic (A2)"]:
    print(n); print(classification_report(yte, preds[n], target_names=CLASSES, digits=3))
"""),
(M, """
## 5. Accuracy by environment
**5a. Simulated conditions** on the real test set (dark, overexposed, motion blur, noise, low resolution, rotation, occlusion).
**5b. Held-out synthetic clips** (stress-test generator), grouped by the environment in the prompt.
**5c. New real photos** per environment, if `test_environments/` exists (recommended: 20–30 photos per class per environment).
"""),
(C, """
cond_rows = {}
for n, m in models_.items():
    cond_rows[n] = {}
    for c in CONDITIONS:
        Xc = np.stack([perturb(x, c, seed=i) for i, x in enumerate(Xte)])
        cond_rows[n][c] = accuracy_score(yte, predict(m, Xc))
cond_tab = pd.DataFrame(cond_rows).T.round(3)
print(cond_tab.to_markdown())

plt.figure(figsize=(11, 3.2))
sns.heatmap(cond_tab, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.3, vmax=1)
plt.title("Accuracy under simulated conditions (real test set)")
plt.tight_layout(); plt.savefig(FIG / "acc_by_condition.png", dpi=150); plt.show()

# example of every condition
fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(2.2 * len(CONDITIONS), 2.6))
for ax, c in zip(axes, CONDITIONS):
    ax.imshow(perturb(Xte[0], c)); ax.set_title(c, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / "conditions_examples.png", dpi=150); plt.show()
"""),
(C, """
stress = syn[syn.split == "stress"].copy()
for n in ["MobileNetV2 real (A1)", "MobileNetV2 + synthetic (A2)"]:
    stress[n] = predict(models_[n], Xstr) == ystr
syn_env_tab = stress.groupby("env")[["MobileNetV2 real (A1)", "MobileNetV2 + synthetic (A2)"]].mean().round(3)
syn_env_tab.loc["ALL"] = stress[["MobileNetV2 real (A1)", "MobileNetV2 + synthetic (A2)"]].mean().round(3)
syn_env_tab["frames"] = stress.groupby("env").size().reindex(syn_env_tab.index).fillna(len(stress)).astype(int)
print("Per-frame accuracy on held-out SYNTHETIC clips:")
print(syn_env_tab.to_markdown())

# clip-level majority vote vs single frame
vote = stress.groupby("clip").apply(lambda d: pd.Series({
    "single-frame acc": d["MobileNetV2 + synthetic (A2)"].mean(),
    "clip vote correct": float(np.bincount(predict(models_["MobileNetV2 + synthetic (A2)"],
                          load_images(d.path)), minlength=3).argmax() == d.label.iloc[0])}))
print("\\nClip-level vote accuracy:", vote["clip vote correct"].mean().round(3))
"""),
(C, """
real_env_tab = None
if ENV_DIR.exists():
    rows = []
    for env in sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir()):
        paths, ys = [], []
        for i, c in enumerate(CLASSES):
            for f in sorted((ENV_DIR / env / c).glob("*")):
                if f.suffix.lower() in EXT: paths.append(str(f)); ys.append(i)
        if not paths: continue
        Xe = load_images(paths); ye = np.array(ys)
        r = {"environment": env, "images": len(ye)}
        for n, m in models_.items(): r[n] = accuracy_score(ye, predict(m, Xe))
        rows.append(r)
    real_env_tab = pd.DataFrame(rows).set_index("environment").round(3)
    print(real_env_tab.to_markdown())
else:
    print("No test_environments/ folder: skipped (optional but recommended).")
"""),
(M, "## 6. Explainability: Grad-CAM"),
(C, """
def cam_grid(idx_list, fname, title):
    names = ["MobileNetV2 real (A1)", "MobileNetV2 + synthetic (A2)"]
    fig, axes = plt.subplots(len(idx_list), 3, figsize=(8, 2.8 * len(idx_list)), squeeze=False)
    for r, i in enumerate(idx_list):
        axes[r, 0].imshow(Xte[i]); axes[r, 0].set_title(f"true: {CLASSES[yte[i]]}", fontsize=9)
        for k, n in enumerate(names):
            cam, probs, ci = grad_cam(models_[n], Xte[i])
            axes[r, k + 1].imshow(overlay_cam(Xte[i], cam))
            axes[r, k + 1].set_title(f"{n.split(' (')[0]}\\n{CLASSES[ci]} {probs[ci]:.0%}", fontsize=8)
        for a in axes[r]: a.axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.show()

rng = np.random.default_rng(SEED)
examples = [int(rng.choice(np.where(yte == c)[0])) for c in range(3)]
cam_grid(examples, "gradcam_examples.png", "Grad-CAM: one test image per class")

wrong = np.where(preds["MobileNetV2 + synthetic (A2)"] != yte)[0][:4]
if len(wrong): cam_grid(list(wrong), "gradcam_errors.png", "Grad-CAM on misclassified test images")

# Grad-CAM on a dark image
cam, probs, ci = grad_cam(models_["MobileNetV2 + synthetic (A2)"], perturb(Xte[examples[0]], "dark"))
plt.figure(figsize=(3, 3)); plt.imshow(overlay_cam(perturb(Xte[examples[0]], "dark"), cam))
plt.title(f"dark: {CLASSES[ci]} {probs[ci]:.0%}"); plt.axis("off"); plt.show()
"""),
(M, "## 7. Latency"),
(C, """
def latency(m, device, n=100):
    x = Xte[:1]
    with tf.device(device):
        for _ in range(10): m(x, training=False)        # warm-up
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); m(x, training=False).numpy(); ts.append((time.perf_counter() - t0) * 1000)
    return np.mean(ts), np.percentile(ts, 95)

lat_rows = []
devices = ["/CPU:0"] + (["/GPU:0"] if tf.config.list_physical_devices("GPU") else [])
for n in ["Baseline CNN (A1)", "MobileNetV2 + synthetic (A2)"]:
    for d in devices:
        mean, p95 = latency(models_[n], d)
        lat_rows.append({"Model": n, "Device": d.strip("/:0"), "Mean ms": round(mean, 1),
                         "p95 ms": round(p95, 1), "FPS": round(1000 / mean, 1),
                         "Params": models_[n].count_params()})
# Grad-CAM cost
t0 = time.perf_counter()
for i in range(20): grad_cam(models_["MobileNetV2 + synthetic (A2)"], Xte[i])
gc_ms = (time.perf_counter() - t0) / 20 * 1000
lat_tab = pd.DataFrame(lat_rows)
print(lat_tab.to_markdown(index=False))
print(f"Grad-CAM (prediction + explanation): {gc_ms:.1f} ms per image")
print("Environment:", platform.platform(), "| CPU:", platform.processor() or "n/a",
      "| GPU:", tf.config.list_physical_devices("GPU"))
"""),
(M, "## 8. Save results and export for the app"),
(C, """
# choose the deployed model by VALIDATION macro-F1 (the test set is not used for model choice)
val_f1 = {n: metrics(yva, predict(models_[n], Xva))["Macro-F1"]
          for n in ["MobileNetV2 real (A1)", "MobileNetV2 + synthetic (A2)"]}
best = max(val_f1, key=val_f1.get)
print("Validation macro-F1:", val_f1, "→ deploy:", best)

APP = OUT / "app_export"; (APP / "examples" / "synthetic").mkdir(parents=True, exist_ok=True)
models_[best].save(APP / "gesture_model.keras")

# a few held-out synthetic clips for the app's "Synthetic clips" tab, watermarked
import cv2
for clip in stress.drop_duplicates("clip").groupby("cls").head(3)["clip"]:
    cls = clip.split("__")[0]
    frames = read_video_frames(CLIPS_DIR / cls / f"{clip}.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(APP / "examples" / "synthetic" / f"{clip}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
    for fr in frames: vw.write(cv2.cvtColor(watermark_frame(fr), cv2.COLOR_RGB2BGR))
    vw.release()

md = ["## Results (test set, 97 real images)", main_tab.to_markdown(), "",
      "## Accuracy under simulated conditions", cond_tab.to_markdown(), "",
      "## Held-out synthetic clips (per-frame accuracy)", syn_env_tab.to_markdown(), "",
      "## Latency", lat_tab.to_markdown(index=False), f"\\nGrad-CAM: {gc_ms:.1f} ms/image"]
if real_env_tab is not None:
    md += ["", "## New real photos by environment", real_env_tab.to_markdown()]
md += ["", "## Dataset", summary.to_markdown(), f"\\nSynthetic share of training set: {syn_share:.1%}"]
(OUT / "results.md").write_text("\\n".join(md)); shutil.copy(OUT / "results.md", APP / "results.md")
main_tab.to_csv(OUT / "main_metrics.csv"); cond_tab.to_csv(OUT / "acc_by_condition.csv")
json.dump({k: {kk: (vv if isinstance(vv, int) else [float(x) for x in vv]) for kk, vv in h.items()}
           for k, h in hists.items()}, open(OUT / "histories.json", "w"))
print("\\n".join(md))
print("\\nSaved to", OUT, "→ copy app_export/* into the app/ folder of the repo.")
"""),
(C, """
# download the app export as a zip
shutil.make_archive("/content/app_export", "zip", APP)
from google.colab import files
files.download("/content/app_export.zip")
"""),
]

(ROOT / "notebooks" / "A2_generate_clips.ipynb").write_text(json.dumps(nb(gen), indent=1))
(ROOT / "notebooks" / "A2_main.ipynb").write_text(json.dumps(nb(main), indent=1))
print("ok")
