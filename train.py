import torch
import comet_ml
from pathlib import Path
from dotenv import dotenv_values
from ultralytics import YOLO

project_name = "yolo26m-hands"

dotenv_path: Path = Path(".env")

logged_in = False
if dotenv_path.exists():
    vals = dotenv_values(dotenv_path)
    api_key = vals.get("COMET_API_KEY")

    if api_key:
        comet_ml.login(api_key=api_key, project_name=project_name)
        logged_in = True

print(f"Comet login:", "OK" if logged_in else "SKIPPED (no `COMET_API_KEY` in `.env` file)")

# device = [-1, -1, -1]
device = (-1 if torch.cuda.is_available() else "cpu")
print(f"Selected device:", "most idle GPU" if device == -1 else "CPU")

model = YOLO("yolo26m-pose.pt")

results = model.train(
    data="hand-keypoints.yaml",
    epochs=500,
    imgsz=640,
    batch=1 * 64,
    patience=5,
    device=device,
    workers=6,
)
