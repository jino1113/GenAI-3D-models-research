# hy3dpaint/src/data/folder_text_dm.py
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

class ImageCaptionDataset(Dataset):
    def __init__(self, root, captions_file=None, image_column="image", caption_column="text",
                 resolution=512, center_crop=True, random_flip=True):
        self.root = Path(root)
        self.items = []

        if captions_file:
            df = pd.read_csv(self.root / captions_file) if captions_file.endswith(".csv") else pd.read_json(self.root / captions_file, lines=True)
            for _, r in df.iterrows():
                self.items.append((self.root / str(r[image_column]), str(r[caption_column])))
        else:
            # ถ้าไม่มีไฟล์คำบรรยาย: ใช้ชื่อไฟล์เป็นแคปชันชั่วคราว
            for p in sorted(self.root.rglob("*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    self.items.append((p, p.stem))

        tfm = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            tfm.append(transforms.CenterCrop(resolution))
        if random_flip:
            tfm.append(transforms.RandomHorizontalFlip())
        tfm += [transforms.ToTensor()]   # [0,1]
        self.transform = transforms.Compose(tfm)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, caption = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"pixel_values": img, "text": caption}

class FolderCaptionDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, captions_file=None, image_column="image", caption_column="text",
                 resolution=512, center_crop=True, random_flip=True, batch_size=2, num_workers=4):
        super().__init__()
        self.kw = dict(root=train_dir, captions_file=captions_file,
                       image_column=image_column, caption_column=caption_column,
                       resolution=resolution, center_crop=center_crop, random_flip=random_flip)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = ImageCaptionDataset(**self.kw)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)
