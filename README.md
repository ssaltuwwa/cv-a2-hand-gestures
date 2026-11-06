# Hand Gesture Recognition: CV Assignment 2

**SE-2403 · Aruzhan Rysbayeva, Alzira Kuanyshbekova, Saltanat Sergeyeva**

A2 continues our A1 project (peace-sign / thumbs-up / stop, 644 self-collected photos) by adding SOTA synthetic motion data: short **text-to-video** clips (Step-Video-T2V idea), cut into frames for training and for stress-testing. It also adds a web demo with Grad-CAM, an accuracy-by-environment evaluation and a latency report.

```
notebooks/
  A2_generate_clips.ipynb   step 1: text-to-video clip generation (Wan2.1-1.3B / Step-Video via Replicate / manual)
  A2_main.ipynb             step 2: frames → training → evaluation → Grad-CAM → latency → export
docs/                       WEB APP (GitHub Pages): the model runs in the browser with ONNX Runtime Web
  index.html, model/gesture.onnx, model/head.json, examples/*.mp4 (SYNTHETIC), results.md
app/                        same app in Python/Gradio, for running locally (python app.py)
  app.py, gesture_utils.py, requirements.txt
scripts/
  build_notebooks.py        regenerates the notebooks (shared code comes from app/gesture_utils.py)
  build_web.py              app_export/ -> docs/ (Keras -> ONNX, clips -> H.264)
  smoke_test.py             runs the whole main notebook on a tiny fake dataset (CPU, ~3 min)
DEMO_SCRIPT.md              plan of the 3–5 min demo video
```

## How to reproduce

**Data (Google Drive):**
```
MyDrive/comp-vis1/
  hand_gestures/{peace-sign,thumbs-up,stop}/        A1 dataset
  a2/synthetic_clips/<class>/*.mp4                  created by A2_generate_clips.ipynb
  a2/test_environments/<env>/<class>/*.jpg          optional: new real photos per environment
```

1. **Generate clips.** Open `A2_generate_clips.ipynb` in Colab (GPU L4/A100) or on Kaggle. Choose `GENERATOR` and run all cells. You get 36 clips (12 per class). Running one clip with Wan2.1-1.3B takes a few minutes. The cell skips finished clips, so it can be re-run after a disconnect.
2. **Train and evaluate.** Open `A2_main.ipynb` in Colab with a GPU runtime and run all cells. In section 2.1, look at the contact sheet and list clips with wrong hands in `REJECT_CLIPS`. Outputs go to `a2/outputs/`: models, `figures/*.png`, `results.md`, CSV tables and `app_export.zip`.
3. **Web app (main demo).** Open the GitHub Pages link of this repository, or serve `docs/` locally:
   `cd docs && python -m http.server` → http://localhost:8000. To rebuild it from a new model:
   `python scripts/build_web.py <app_export folder> docs` (needs tensorflow, tf2onnx, onnxruntime, ffmpeg).
4. **Python app (optional).**
   ```bash
   cd app
   # unzip app_export.zip here → gesture_model.keras, results.md, examples/synthetic/*.mp4
   pip install -r requirements.txt     # set tensorflow-cpu==2.21.0 is the Colab version
   python app.py                       # http://127.0.0.1:7860
   ```

**Environment used:** Google Colab (Python 3.13, TensorFlow 2.21.0, Tesla T4 GPU), seed 42. The split is identical to A1.

## Synthetic media and ethics
All generated clips are AI-made and contain no real people. In the app every synthetic preview carries a visible `SYNTHETIC - AI-generated` label. Training frames are labelled by folder and file name instead of a watermark, so the model cannot use the watermark as a shortcut. All people in the real photos are team members who agreed to be photographed. The web app runs the model in the browser, so photos and webcam frames never leave the device.

## Credits
- Step-Video-T2V: Stepfun AI, arXiv:2502.10248; Neurohive overview: https://neurohive.io/ru/papers/step-video-t2v-generator-video-iz-teksta-s-koefficientom-szhatiya-16x/
- Wan2.1: Wan-AI (open T2V model used as the runnable lighter alternative)
- MobileNetV2 (ImageNet weights, Keras Applications); Grad-CAM (Selvaraju et al., 2017)
