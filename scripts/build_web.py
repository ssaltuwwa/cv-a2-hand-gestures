"""Build the static web app (GitHub Pages) from the notebook's app_export folder.

Usage: python scripts/build_web.py path/to/app_export docs
  - converts gesture_model.keras -> web/model/gesture.onnx + head.json
  - re-encodes the watermarked synthetic clips to H.264 (browsers cannot play mp4v) -> web/examples/
  - copies results.md
"""
import sys, json, shutil, subprocess
from pathlib import Path

src, web = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).parent
subprocess.run([sys.executable, str(here / "export_web_model.py"), str(src / "gesture_model.keras"),
                str(web / "model")], check=True)
ex = web / "examples"; ex.mkdir(parents=True, exist_ok=True)
names = []
for clip in sorted((src / "examples" / "synthetic").glob("*.mp4")):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(ex / clip.name)], check=True)
    names.append(clip.name)
json.dump(names, open(ex / "clips.json", "w"), indent=1)
shutil.copy(src / "results.md", web / "results.md")
print("clips:", len(names), "| web app ready in", web)
