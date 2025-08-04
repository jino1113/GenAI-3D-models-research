import torch
import numpy as np
import trimesh
import os
from my_dataset import My3DDataset
from train_hunyuan import Hunyuan3DModel

# ---- สร้างโฟลเดอร์สำหรับเก็บไฟล์ที่ export ----
output_dir_pred = "exported_objs/predicted"
output_dir_gt = "exported_objs/groundtruth"
os.makedirs(output_dir_pred, exist_ok=True)
os.makedirs(output_dir_gt, exist_ok=True)

# ---- โหลดโมเดล ----
model = Hunyuan3DModel()
model.load_state_dict(torch.load("hunyuan3d_model.pth"))
model.eval()

# ---- ฟังก์ชันแปลง voxel เป็น mesh ----
def voxel_to_mesh(voxel_tensor, filename="output.obj", threshold=0.5):
    # แปลง voxel tensor -> numpy array
    voxel = voxel_tensor.squeeze().cpu().numpy()
    voxel_binary = (voxel > threshold).astype(np.float32)

    # ใช้ marching cubes คืนเป็น Trimesh object
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_binary)

    # Export mesh เป็น .obj
    mesh.export(filename)
    print(f"Saved: {filename}")

# ---- โหลด dataset ----
dataset = My3DDataset()

# ---- Export ทั้ง dataset ----
with torch.no_grad():
    for idx, (img, voxel_gt) in enumerate(dataset):
        # Predict voxel จากโมเดล
        pred_voxel = model(img.unsqueeze(0))  # [1, 1, 32, 32, 32]

        # ตั้งชื่อไฟล์
        pred_filename = os.path.join(output_dir_pred, f"predicted_{idx}.obj")
        gt_filename = os.path.join(output_dir_gt, f"groundtruth_{idx}.obj")

        # Export ทั้ง predicted และ groundtruth
        voxel_to_mesh(pred_voxel, pred_filename)
        voxel_to_mesh(voxel_gt, gt_filename)
