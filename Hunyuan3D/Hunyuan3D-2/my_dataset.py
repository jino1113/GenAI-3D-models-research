import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from test_voxel import mesh_to_voxel

class My3DDataset(Dataset):
    def __init__(self, img_dir="dataset/images", mesh_dir="dataset/meshes", transform=None):
        self.img_dir = img_dir
        self.mesh_dir = mesh_dir
        self.images = sorted(os.listdir(img_dir))
        self.meshes = sorted(os.listdir(mesh_dir))
        self.transform = transform

        # ทำให้จำนวนภาพและ mesh เท่ากัน (วนซ้ำถ้า mesh น้อยกว่า)
        if len(self.meshes) < len(self.images):
            repeats = (len(self.images) // len(self.meshes)) + 1
            self.meshes = (self.meshes * repeats)[:len(self.images)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # โหลดรูปภาพและแปลงเป็น tensor [3, 64, 64]
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # โหลด voxel grid (แปลงและ resize ใน mesh_to_voxel)
        mesh_path = os.path.join(self.mesh_dir, self.meshes[idx])
        voxel_tensor = mesh_to_voxel(mesh_path, voxel_size=32)  # [1, 32, 32, 32]

        return img_tensor, voxel_tensor
