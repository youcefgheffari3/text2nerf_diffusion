import os
import torch
import numpy as np
import imageio
from models.gaussian_scene import GaussianScene
from utils.camera_utils import create_orbit_camera_path

# === Config ===
OUT_DIR = "outputs/final_render"
CHECKPOINT = "outputs/scene_consistent_0500.pt"  # Change to your final model
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
scene = GaussianScene(num_points=4096, radius=2.5, device=device).to(device)
scene.load_state_dict(torch.load(CHECKPOINT))
scene.eval()

# === Generate circular camera path
camera_path = create_orbit_camera_path(num_frames=60, radius=2.5)

# === Render loop
print("🎥 Rendering...")
frames = []
for i, cam in enumerate(camera_path):
    intr, extr = cam['K'], cam['pose']
    image = scene.render(extr, intr, image_size=(512, 512))[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    frames.append(image)
    imageio.imwrite(os.path.join(OUT_DIR, f"frame_{i:03d}.png"), image)

# === Save as video or GIF
video_path = os.path.join(OUT_DIR, "orbit.mp4")
imageio.mimsave(video_path, frames, fps=15)
print(f"✅ Render complete: {video_path}")

