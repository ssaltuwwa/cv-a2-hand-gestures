"""Hand Gesture Recognition - A2 demo (Gradio).

Tabs: Predict (upload / webcam) · Live webcam · Stress test · Synthetic clips · About & ethics.
Run locally:  pip install -r requirements.txt && python app.py
"""
import os
import time
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf

from gesture_utils import (CLASSES, CONDITIONS, to_model_input, perturb, grad_cam,
                           overlay_cam, read_video_frames, watermark_frame)

HERE = Path(__file__).parent
MODEL_PATH = HERE / os.environ.get("MODEL_FILE", "gesture_model.keras")
SYN_DIR = HERE / "examples" / "synthetic"
RESULTS_MD = HERE / "results.md"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
_ = model(np.zeros((1, 224, 224, 3), np.float32), training=False)   # warm-up

GPU = tf.config.list_physical_devices("GPU")
SERVER_ENV = (f"Server: {platform.system()} {platform.machine()} · "
              f"{'GPU' if GPU else 'CPU'} · Python {platform.python_version()} · "
              f"TensorFlow {tf.__version__}")


def browser_from_ua(ua: str) -> str:
    ua = ua or ""
    for name in ("Edg", "OPR", "Chrome", "Firefox", "Safari"):
        if name in ua:
            br = {"Edg": "Edge", "OPR": "Opera"}.get(name, name)
            break
    else:
        br = "unknown browser"
    os_ = next((o for o in ("Android", "iPhone", "iPad", "Windows", "Mac OS", "Linux")
                if o in ua), "unknown OS")
    return f"{br} on {os_}"


def predict_probs(img01):
    t0 = time.perf_counter()
    p = model(img01[None], training=False).numpy()[0]
    return p, (time.perf_counter() - t0) * 1000


def env_text(request, last_ms, history):
    ua = request.headers.get("user-agent", "") if request else ""
    avg = np.mean(history) if history else last_ms
    return (f"**Inference:** {last_ms:.1f} ms (session avg {avg:.1f} ms over {len(history)} runs)  \n"
            f"**Client:** {browser_from_ua(ua)}  \n**{SERVER_ENV}**")


# ------------------------------------------------------------- Predict tab
def run_predict(image, show_cam, history, request: gr.Request):
    if image is None:
        return None, None, "Upload a photo or take one with the webcam.", history
    x = to_model_input(image)
    probs, ms = predict_probs(x)
    history = (history or []) + [ms]
    labels = {c: float(p) for c, p in zip(CLASSES, probs)}
    cam_img = None
    if show_cam:
        cam, _, idx = grad_cam(model, x)
        cam_img = overlay_cam(x, cam)
    return labels, cam_img, env_text(request, ms, history), history


# --------------------------------------------------------------- Live tab
def run_live(frame, history, request: gr.Request):
    if frame is None:
        return None, "", history
    x = to_model_input(frame)
    probs, ms = predict_probs(x)
    history = ((history or []) + [ms])[-200:]
    best = int(np.argmax(probs))
    label = CLASSES[best] if probs[best] >= 0.60 else "unsure"
    return ({c: float(p) for c, p in zip(CLASSES, probs)},
            f"### {label}\n" + env_text(request, ms, history), history)


# -------------------------------------------------------- Stress-test tab
def run_stress(image, conditions):
    if image is None:
        return [], None
    conditions = conditions or CONDITIONS
    x = to_model_input(image)
    gallery, rows = [], []
    for c in conditions:
        xc = perturb(x, c, seed=0)
        probs, ms = predict_probs(xc)
        k = int(np.argmax(probs))
        gallery.append((np.uint8(xc * 255), f"{c}: {CLASSES[k]} {probs[k]*100:.0f}%"))
        rows.append({"condition": c, "prediction": CLASSES[k],
                     **{f"P({cl})": round(float(p), 3) for cl, p in zip(CLASSES, probs)},
                     "latency_ms": round(ms, 1)})
    return gallery, pd.DataFrame(rows)


# ------------------------------------------------------ Synthetic clips tab
def list_clips():
    return sorted(p.name for p in SYN_DIR.glob("*.mp4"))


def run_clip(name):
    if not name:
        return None, None, ""
    path = SYN_DIR / name
    frames = read_video_frames(path, n=24)
    probs = np.stack([predict_probs(to_model_input(f))[0] for f in frames])
    votes = probs.argmax(1)
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, c in enumerate(CLASSES):
        ax.plot(probs[:, i], label=c, marker="o", ms=3)
    ax.set_xlabel("frame"); ax.set_ylabel("probability"); ax.set_ylim(0, 1)
    ax.set_title("Per-frame prediction (SYNTHETIC clip)"); ax.legend(loc="upper right")
    fig.tight_layout()
    counts = {c: int((votes == i).sum()) for i, c in enumerate(CLASSES)}
    maj = max(counts, key=counts.get)
    true = next((c for c in CLASSES if name.startswith(c)), None)
    txt = (f"**Majority vote:** {maj} ({counts[maj]}/{len(frames)} frames)  \n"
           f"**Frame votes:** {counts}  \n"
           + (f"**Prompted gesture:** {true} → per-frame accuracy "
              f"{(votes == CLASSES.index(true)).mean()*100:.0f}%" if true else ""))
    preview = watermark_frame(frames[len(frames) // 2])
    return preview, fig, txt


def about_md():
    md = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else \
        "_Results table will appear here (copy `results.md` from the notebook outputs)._"
    return md


ETHICS = """
**Privacy & consent.** Photos you upload or capture are processed in memory and are **not stored**.
All people in our training photos are team members who agreed to be photographed for this course project.

**Synthetic media.** Clips in the *Synthetic clips* tab are **AI-generated** (text-to-video) and are
labelled `SYNTHETIC` on every preview. They were used only to augment training data and to stress-test
the classifier; they do not depict real people.

**Limitations.** Trained on ~650 photos of a few people; accuracy drops in conditions not covered by the
data (see stress test). Not suitable for safety-critical or surveillance use.
"""

with gr.Blocks(title="Hand Gesture Recognition - A2") as demo:
    gr.Markdown("# ✋ Hand Gesture Recognition: peace · thumbs-up · stop\n"
                "MobileNetV2 fine-tuned on our own photos (chosen by validation F1). "
                "AI-generated (synthetic) clips are used for training experiments and stress tests. "
                "Images are processed in memory and not stored.")
    hist = gr.State([])

    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(sources=["upload", "webcam"], type="pil", label="Photo")
                cam_cb = gr.Checkbox(True, label="Show Grad-CAM explanation")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                out_lbl = gr.Label(num_top_classes=3, label="Prediction")
                out_cam = gr.Image(label="Grad-CAM: where the model looks")
                out_env = gr.Markdown()
        btn.click(run_predict, [inp, cam_cb, hist], [out_lbl, out_cam, out_env, hist])

    with gr.Tab("Live webcam"):
        gr.Markdown("Show a gesture to the camera. Below 60% confidence the app says *unsure*.")
        with gr.Row():
            live_in = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
            with gr.Column():
                live_lbl = gr.Label(num_top_classes=3)
                live_env = gr.Markdown()
        live_in.stream(run_live, [live_in, hist], [live_lbl, live_env, hist],
                       stream_every=0.5, time_limit=120)

    with gr.Tab("Stress test"):
        gr.Markdown("Simulate difficult environments on your own photo and see how the prediction changes.")
        with gr.Row():
            with gr.Column():
                s_in = gr.Image(sources=["upload", "webcam"], type="pil", label="Photo")
                s_cond = gr.CheckboxGroup(CONDITIONS, value=CONDITIONS, label="Conditions")
                s_btn = gr.Button("Run stress test", variant="primary")
            s_gal = gr.Gallery(label="Simulated conditions", columns=4, height="auto")
        s_tab = gr.Dataframe(label="Predictions per condition")
        s_btn.click(run_stress, [s_in, s_cond], [s_gal, s_tab])

    with gr.Tab("Synthetic clips"):
        gr.Markdown("⚠️ **All clips here are AI-generated (SYNTHETIC).** Generated offline in Colab, "
                    "used as a stress-test generator: the model is run on every frame.")
        clip_dd = gr.Dropdown(list_clips(), label="Choose a generated clip")
        with gr.Row():
            clip_img = gr.Image(label="Middle frame (watermarked)")
            clip_plot = gr.Plot(label="Per-frame probabilities")
        clip_txt = gr.Markdown()
        clip_dd.change(run_clip, clip_dd, [clip_img, clip_plot, clip_txt])

    with gr.Tab("About & ethics"):
        gr.Markdown(ETHICS)
        gr.Markdown(about_md())

if __name__ == "__main__":
    demo.launch()
