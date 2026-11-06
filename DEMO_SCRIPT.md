# Demo video script (≈4 min)

Record the screen with OBS or the built-in recorder. One person narrates in English.

| Time | What is on screen | What to say |
|---|---|---|
| 0:00–0:20 | Title slide / app header | "Hi, we are … from SE-2403. In A1 we built a hand gesture classifier for peace sign, thumbs up and stop. In A2 we added SOTA synthetic motion data and built a web demo." |
| 0:20–0:50 | `A2_generate_clips.ipynb`: prompts cell, 2–3 generated clips | "Following the Step-Video-T2V idea, we generated short clips from text prompts: gesture × environment × camera motion. These clips are AI-generated, so we label them as synthetic everywhere." |
| 0:50–1:10 | Contact sheet with rejected clips | "Text-to-video models often draw hands wrongly, so we checked every clip by hand and removed N of 36." |
| 1:10–1:50 | Results tables + condition heatmap | "On the real test set, accuracy went from X to Y. The biggest gains are in dark and motion-blur conditions. The ablation without augmentation shows …" |
| 1:50–2:40 | App → Predict tab: upload 3 photos, then webcam | "Here is the prediction with a Grad-CAM heatmap: the model looks at the fingers, not the background. Latency is shown here: Z ms on CPU." |
| 2:40–3:10 | Live webcam tab | Show all three gestures live, then an ambiguous hand → "unsure". |
| 3:10–3:40 | Stress test tab | "Our stress-test generator simulates difficult conditions on any photo. You can see where the model breaks: for example heavy occlusion." |
| 3:40–4:00 | Synthetic clips tab + About & ethics | "Generated clips are watermarked, and uploaded photos are not stored. Limitations: few people and a small test set. Thank you!" |
