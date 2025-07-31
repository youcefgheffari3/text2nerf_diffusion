import numpy as np
import torch

def create_random_cameras(num_views=4, radius=2.5):
    cameras = []
    for i in range(num_views):
        theta = np.pi * 2 * i / num_views
        cam_pos = np.array([radius * np.cos(theta), 0.0, radius * np.sin(theta)])

        # Look-at center
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        z = (cam_pos - target)
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        # Camera-to-world matrix
        pose = np.eye(4)
        pose[:3, :3] = np.stack([x, y, z], axis=0)
        pose[:3, 3] = cam_pos

        # Intrinsics (assuming fx = fy = 500, cx = cy = 256)
        K = np.array([[500.0, 0, 256.0],
                      [0, 500.0, 256.0],
                      [0, 0, 1]])

        cameras.append({
            'K': K,
            'pose': pose
        })
    return cameras
