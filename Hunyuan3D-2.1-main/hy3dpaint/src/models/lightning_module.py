# hy3dpaint/src/models/lightning_module.py
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

# ใช้ wrapper นี้กับ HunyuanPaintPipeline (หรือ StableDiffusionPipeline) ที่สร้างไว้แล้ว
class HunyuanPaintLightningModule(pl.LightningModule):
    def __init__(self, pipeline, learning_rate: float = 5e-5, weight_decay: float = 1e-2):
        super().__init__()
        self.save_hyperparameters(ignore=["pipeline"])
        self.pipeline = pipeline
        self.lr = learning_rate
        self.wd = weight_decay

        # Freeze ส่วนที่ไม่ train
        for p in self.pipeline.vae.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder.parameters():
            p.requires_grad = False

        # เอา UNet มาเป็นโมเดลหลักที่ train
        self.unet = self.pipeline.unet

        # ใช้ DDPMScheduler สำหรับการ add_noise ตอน train
        cfg = self.pipeline.scheduler.config
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=getattr(cfg, "num_train_timesteps", 1000),
            beta_start=getattr(cfg, "beta_start", 0.00085),
            beta_end=getattr(cfg, "beta_end", 0.012),
            beta_schedule=getattr(cfg, "beta_schedule", "scaled_linear"),
        )

        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder

    def configure_optimizers(self):
        # train เฉพาะพารามิเตอร์ที่ requires_grad=True (ส่วนใหญ่คือ UNet/LoRA)
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        return opt

    def _get_images(self, batch: Dict[str, Any]) -> torch.Tensor:
        # รองรับหลาย key ที่ dataset อาจส่งมา
        if "pixel_values" in batch:
            x = batch["pixel_values"]
            # assume x ∈ [0,1] → map ไป [-1,1] ถ้ายังไม่ได้ทำ
            if x.min() >= 0 and x.max() <= 1:
                x = x * 2 - 1
            return x
        for k in ["images", "image", 0]:
            if k in batch:
                x = batch[k]
                if x.min() >= 0 and x.max() <= 1:
                    x = x * 2 - 1
                return x
        raise KeyError("Dataset batch must contain key 'pixel_values' or 'images'/'image'.")

    def _get_texts(self, batch: Dict[str, Any]) -> List[str]:
        for k in ["text", "caption", "captions", 1]:
            if k in batch:
                return batch[k]
        # ถ้าไม่มี caption ให้ใส่คำสั้น ๆ ป้องกันหลุด
        bsz = next(iter(batch.values())).shape[0]
        return [""] * bsz

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        device = self.device
        dtype = self.unet.dtype

        # 1) ภาพ → latents
        images = self._get_images(batch).to(device=device, dtype=self.vae.dtype)  # [-1, 1]
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        latents = latents.to(device=device, dtype=dtype)

        # 2) ข้อความ → prompt embeds
        texts = self._get_texts(batch)
        with torch.no_grad():
            tok = self.tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            prompt_embeds = self.text_encoder(**tok)[0].to(dtype)

        # 3) sample noise + timestep
        noise = randn_tensor(latents.shape, device=device, dtype=dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 4) predict noise ด้วย UNet
        model_out = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds)
        pred = getattr(model_out, "sample", model_out)

        # 5) loss = MSE(pred, noise)
        loss = F.mse_loss(pred.float(), noise.float())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
