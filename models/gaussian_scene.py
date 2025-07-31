import torch
import torch.nn as nn
import numpy as np

class GaussianScene(nn.Module):
    def __init__(self, num_points=4096, radius=2.5, device="cuda"):
        super().__init__()
        self.device = device

        # 3D point positions in spherical shell around origin
        phi = torch.rand(num_points) * 2 * np.pi
        costheta = torch.rand(num_points) * 2 - 1
        u = torch.rand(num_points)

        theta = torch.acos(costheta)
        r = radius * (u ** (1/3))

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        # Learnable point positions (3D)
        self.positions = nn.Parameter(torch.stack([x, y, z], dim=-1).to(device))

        # Learnable colors (RGB)
        self.colors = nn.Parameter(torch.rand(num_points, 3).to(device))

        # Optional: learnable opacity or scale (not mandatory here)

    def render(self, camera_pose, intrinsics, image_size=(512, 512)):
        """
        Projects 3D points to the 2D image plane using camera parameters.
        Returns: Rendered RGB image (1, 3, H, W)
        """
        B, H, W = 1, image_size[0], image_size[1]

        # Extract R (3x3) and t (3x1)
        R = torch.tensor(camera_pose[:3, :3], dtype=torch.float32).to(self.device)
        t = torch.tensor(camera_pose[:3, 3:], dtype=torch.float32).to(self.device)

        # Project points
        points_cam = (R @ self.positions.T).T + t.T  # [N, 3]

        # Intrinsics
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

        # Perspective projection
        u = fx * x / z + cx
        v = fy * y / z + cy

        # Create blank image
        img = torch.zeros((B, 3, H, W), device=self.device)

        for i in range(self.positions.shape[0]):
            xi = int(u[i].item())
            yi = int(v[i].item())
            if 0 <= xi < W and 0 <= yi < H:
                img[0, :, yi, xi] = self.colors[i]

        return img.clamp(0, 1)
