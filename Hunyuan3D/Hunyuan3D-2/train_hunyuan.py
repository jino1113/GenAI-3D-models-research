import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from my_dataset import My3DDataset

# โหลด Dataset
dataset = My3DDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# โมเดล 2D -> 3D
class Hunyuan3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64*64*3, 32*32*32)  # map image to voxel

    def forward(self, x):
        # บังคับให้ tensor contiguous ก่อน flatten
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), 1, 32, 32, 32)

    def compute_loss(self, pred, target):
        return nn.functional.mse_loss(pred, target)

model = Hunyuan3DModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# เทรน 5 epochs
for epoch in range(5):
    for images, voxels in dataloader:
        preds = model(images)
        loss = model.compute_loss(preds, voxels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), "hunyuan3d_model.pth")
print("Model saved as hunyuan3d_model.pth")
