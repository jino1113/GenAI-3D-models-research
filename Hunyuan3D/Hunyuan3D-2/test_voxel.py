import trimesh
import torch
import numpy as np
import torch.nn.functional as F

def mesh_to_voxel(mesh_path, voxel_size=32):
    mesh = trimesh.load(mesh_path)

    # ถ้าโหลดออกมาเป็น Scene (รวมหลาย Mesh) ให้รวมทั้งหมด
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # voxelization (VoxelGrid)
    pitch = mesh.extents.max() / voxel_size
    voxelized = mesh.voxelized(pitch=pitch)

    # เอา voxel matrix (filled volume) ออกมา
    if hasattr(voxelized, 'matrix'):
        filled = voxelized.matrix
    else:
        filled = voxelized.fill()  # fallback

    # แปลงเป็น torch tensor [1, 1, D, H, W]
    tensor = torch.from_numpy(np.array(filled, dtype=np.float32)).unsqueeze(0).unsqueeze(0)

    # Resize voxel ให้ขนาดคงที่ [1, 1, 32, 32, 32] สำหรับโมเดล
    tensor_resized = F.interpolate(tensor, size=(32, 32, 32), mode="trilinear", align_corners=False)

    print("Voxel shape after resize:", tensor_resized.shape)  # Debug
    return tensor_resized.squeeze(0)  # [1, 32, 32, 32]

# ทดสอบแปลง mesh
if __name__ == "__main__":
    voxel = mesh_to_voxel(
        r"C:\Users\Jino\Documents\GitHub\GenAI_3D_models_research\Hunyuan3D\Hunyuan3D-2\dataset\meshes\furniture_armchair_teagan.glb"
    )
    print("Final voxel shape for model:", voxel.shape)  # [1, 32, 32, 32]
