import os
import torch
import numpy as np
from tqdm import tqdm
from models.gaussian_scene import GaussianScene
from models.stable_diffusion_sds import StableDiffusionSDS
from utils.camera_utils import create_random_cameras
import torch.nn.functional as F

# === Config ===
prompt = "A futuristic sci-fi city street at sunset, cinematic, highly detailed"
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = (512, 512)
num_iters = 500
lr = 1e-2
num_views = 4
consistency_weight = 1.0

# === Load models ===
scene = GaussianScene(num_points=4096, radius=2.5, device=device).to(device)
sds = StableDiffusionSDS(device=device)

optimizer = torch.optim.Adam(scene.parameters(), lr=lr)

# === Training ===
print("🚀 Starting training with view consistency...")
for step in tqdm(range(num_iters)):
    loss_total = 0.0
    cameras = create_random_cameras(num_views=num_views, radius=2.5)

    rendered_images = []
    for cam in cameras:
        intr, extr = cam['K'], cam['pose']
        rendered = scene.render(extr, intr, image_size=image_size)  # [1,3,H,W]
        rendered_images.append(rendered)

    # === Text guidance (SDS) loss
    for img in rendered_images:
        loss_total += sds(prompt=prompt, image=img)

    # === View consistency loss (L2)
    for i in range(len(rendered_images)):
        for j in range(i+1, len(rendered_images)):
            loss_total += consistency_weight * F.mse_loss(rendered_images[i], rendered_images[j])

    loss_total /= (len(rendered_images) + 1)

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"🧠 Iter {step:04d} | Loss: {loss_total.item():.4f}")
        torch.save(scene.state_dict(), f"outputs/scene_consistent_{step:04d}.pt")

print("✅ Training with view consistency complete.")
