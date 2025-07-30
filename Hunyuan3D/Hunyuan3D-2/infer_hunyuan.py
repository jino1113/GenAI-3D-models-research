import torch
from my_dataset import My3DDataset
from train_hunyuan import Hunyuan3DModel
import numpy as np
import trimesh

# โหลดโมเดล
model = Hunyuan3DModel()
model.load_state_dict(torch.load("hunyuan3d_model.pth"))
model.eval()

# ใช้ภาพแรกจาก dataset
dataset = My3DDataset()
img, voxel_gt = dataset[0]  # (image, ground truth voxel)

# รันอินเฟอร์
with torch.no_grad():
    pred_voxel = model(img.unsqueeze(0))  # [1, 1, 32, 32, 32]

print("Predicted voxel shape:", pred_voxel.shape)
print("Ground truth voxel shape:", voxel_gt.shape)

# เซฟไฟล์ผลลัพธ์
torch.save(pred_voxel, "predicted_voxel.pt")
print("Predicted voxel saved to predicted_voxel.pt")

# ---- Export to OBJ ----
def voxel_to_mesh(voxel_tensor, filename="predicted.obj"):
    voxel = voxel_tensor.squeeze().cpu().numpy()
    # สร้าง mesh จาก voxel
    vertices, faces, normals, _ = trimesh.voxel.ops.matrix_to_marching_cubes(voxel)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.export(filename)
    print(f"Voxel exported as {filename}")

voxel_to_mesh(pred_voxel, "predicted.obj")
