"""Smoke test: run A2_main.ipynb code on a tiny fake dataset (CPU, no ImageNet download, 1 epoch)."""
import json, re, sys, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image

ROOT = Path(sys.argv[1])          # fake drive root
rng = np.random.default_rng(0)
CL = ["peace-sign", "thumbs-up", "stop"]
for c in CL:
    d = ROOT / "hand_gestures" / c; d.mkdir(parents=True, exist_ok=True)
    for i in range(14):
        Image.fromarray(rng.integers(0, 255, (300, 260, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    cd = ROOT / "a2" / "synthetic_clips" / c; cd.mkdir(parents=True, exist_ok=True)
    for k in range(4):
        vw = cv2.VideoWriter(str(cd / f"{c}__dark_room__pan__{k:02d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (208, 120))
        for _ in range(20): vw.write(rng.integers(0, 255, (120, 208, 3), dtype=np.uint8))
        vw.release()
    ed = ROOT / "a2" / "test_environments" / "dark" / c; ed.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(ed / f"{i}.jpg")

nb = json.load(open(Path(__file__).parents[1] / "notebooks" / "A2_main.ipynb"))
code = []
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    s = "".join(c["source"])
    if "google.colab" in s and "drive.mount" in s: continue
    if "files.download" in s: continue
    s = s.replace('Path("/content/drive/MyDrive/comp-vis1")', f'Path("{ROOT}")')
    s = s.replace('weights="imagenet" if imagenet else None', 'weights=None')
    s = re.sub(r"epochs=1[05]", "epochs=1", s)
    s = s.replace("n=100", "n=5").replace("range(20)", "range(3)")
    code.append(s)
import matplotlib; matplotlib.use("Agg")
g = {}
exec(compile("\n\n".join(code), "nb", "exec"), g)
print("SMOKE TEST PASSED")
