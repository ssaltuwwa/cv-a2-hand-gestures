---
title: Hand Gesture Recognition A2
emoji: ✋
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Hand Gesture Recognition: A2 demo

Upload or webcam → gesture (peace-sign / thumbs-up / stop) → Grad-CAM explanation, with latency and device info.
Extra tabs: live webcam, stress test (simulated dark / blur / noise / rotation / occlusion), AI-generated synthetic clips (labelled SYNTHETIC).

Files needed next to `app.py` (exported by `A2_main.ipynb`, section 8): `gesture_model.keras`, `results.md`, `examples/synthetic/*.mp4`.
