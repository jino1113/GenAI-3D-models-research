# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Any, Dict, Optional, Callable, List, Union
import numpy as np
import torch
from PIL import Image
from einops import rearrange

from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline, retrieve_timesteps, rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import deprecate

from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

try:
    from omegaconf import DictConfig
    DictLike = (dict, DictConfig)
except Exception:
    DictLike = (dict,)

# local import for optional 2.5D wrapper
from .unet.modules import UNet2p5DConditionModel

__all__ = ["HunyuanPaintPipeline"]


def to_rgb_image(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        bg = Image.fromarray(np.full((img.height, img.width, 3), 127, np.uint8), "RGB")
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    raise ValueError("Unsupported image type.", img.mode)


class HunyuanPaintPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        feature_extractor: CLIPFeatureExtractor,
        **extra,
    ):
        # 1) read 2.5D flag (default False)
        use_25d = bool(extra.pop("use_25d", False))
        # discard stray kwargs from configs
        extra.pop("kwargs", None)

        # 2) optionally wrap UNet into 2.5D BEFORE super().__init__
        if use_25d and isinstance(unet, UNet2DConditionModel):
            unet = UNet2p5DConditionModel(unet, train_sched=None, infer_sched=scheduler)

        # 3) call parent init
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )

        # 4) basic pipeline buffers
        self.use_25d = use_25d
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # no-op; avoid re-wrapping UNet here
    def prepare(self) -> None:
        return

    def eval(self):
        self.unet.eval()
        self.vae.eval()

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, N, C, H, W) in [0,1]
        returns latents: (B, N, C', H', W')
        """
        B = images.shape[0]
        x = rearrange(images, "b n c h w -> (b n) c h w")
        dtype = next(self.vae.parameters()).dtype
        x = (x - 0.5) * 2.0
        lat = self.vae.encode(x.to(dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        return rearrange(lat, "(b n) c h w -> b n c h w", b=B)

    @torch.no_grad()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = "watermark, ugly, deformed, noisy, blurry, low contrast",
        *args,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 3.0,
        output_type: str = "pil",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 15,
        return_dict: bool = True,
        **cached_condition,
    ):
        # one-time lightweight preparation (no wrapping)
        self.prepare()

        # normalize images input
        if not isinstance(images, list):
            images = [images]
        images = [to_rgb_image(im) for im in images]

        # stack into (1, N, C, H, W)
        imgs = [torch.tensor(np.array(im) / 255.0) for im in images]
        imgs = [t.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0) for t in imgs]
        images_vae = torch.cat(imgs, dim=1).to(self.device, dtype=next(self.vae.parameters()).dtype)

        assert images_vae.shape[0] == 1 and num_images_per_prompt == 1, "Only batch=1 is currently supported."

        # reference encode for RA mode if available
        if getattr(self.unet, "use_ra", False):
            cached_condition["ref_latents"] = self.encode_images(images_vae)

        # prompt setup
        device = self._execution_device
        if prompt is None:
            prompt = "high quality"
        if isinstance(prompt, str):
            prompt = [prompt]
        prompt_embeds, _ = self.encode_prompt(
            prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        negative_prompt_embeds, _ = self.encode_prompt(
            negative_prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )

        # mark CFG usage for downstream helpers
        self.do_classifier_free_guidance = bool(guidance_scale and guidance_scale > 1.0)

        return self.denoise(
            None, *args,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            width=width, height=height,
            return_dict=return_dict,
            **cached_condition,
            num_in_batch=1,
            camera_azims=[0],
        )

    def denoise(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator=None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        kwargs["cache"] = {}

        if callback is not None:
            deprecate("callback", "1.0.0", "Passing `callback` is deprecated; use `callback_on_step_end`.")
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0", "Passing `callback_steps` is deprecated; use `callback_on_step_end`.")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # dimensions
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # input checks (uses self.do_classifier_free_guidance internally)
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt,
            prompt_embeds, negative_prompt_embeds,
            ip_adapter_image, ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # build encoder states for CFG triple pass: [uncond, ref, full]
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds])

        # IP-Adapter
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, ip_adapter_image_embeds, self.device,
                1 * num_images_per_prompt, self.do_classifier_free_guidance,
            )
        else:
            image_embeds = None

        # timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps, sigmas
        )

        # PBR multiplicity (fallback to 1 if the attribute doesn't exist)
        pbr_setting = getattr(self.unet, "pbr_setting", None)
        n_pbr = len(pbr_setting) if pbr_setting is not None else 1

        # latents
        num_channels_latents = self.unet.config.in_channels
        num_in_batch = kwargs.get("num_in_batch", 1)
        latents = self.prepare_latents(
            1 * num_in_batch * n_pbr,
            num_channels_latents, height, width,
            prompt_embeds.dtype, self.device, generator, latents,
        )

        # step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {"image_embeds": image_embeds} if image_embeds is not None else None

        # guidance scale embedding (for sd-xl like models)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            g = torch.tensor(guidance_scale - 1).repeat(num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                g, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # (b, n_pbr, n, c, h, w)
                latents = rearrange(latents, "(b n_pbr n) c h w -> b n_pbr n c h w", n=num_in_batch, n_pbr=n_pbr)
                latent_model_input = latents.repeat(3, 1, 1, 1, 1, 1) if self.do_classifier_free_guidance else latents
                latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = rearrange(
                    latent_model_input, "(b n_pbr n) c h w -> b n_pbr n c h w", n=num_in_batch, n_pbr=n_pbr
                )

                cross_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else self.cross_attention_kwargs

                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False, **kwargs,
                )[0]

                # back to (batch, channels, h, w)
                latents = rearrange(latents, "b n_pbr n c h w -> (b n_pbr n) c h w")

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_ref, noise_pred_full = noise_pred.chunk(3)
                    azims = kwargs.get("camera_azims", [0] * num_in_batch)

                    def cam_map(a: float) -> float:
                        if 0 <= a < 90:
                            return a / 90.0 + 1
                        if 90 <= a < 330:
                            return 2.0
                        return -a / 90.0 + 5.0

                    view_scale = torch.from_numpy(np.asarray([cam_map(a) for a in azims])).unsqueeze(0).repeat(n_pbr, 1).view(-1)
                    view_scale = view_scale.to(noise_pred_uncond)[:, None, None, None]

                    # two-stage CFG: uncond -> ref -> full
                    noise_pred = noise_pred_uncond + guidance_scale * view_scale * (noise_pred_ref - noise_pred_uncond)
                    noise_pred += guidance_scale * view_scale * (noise_pred_full - noise_pred_ref)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # rescale to prevent overexposure
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_ref, guidance_rescale=guidance_rescale)

                latents = self.scheduler.step(
                    noise_pred, t, latents[:, :num_channels_latents, :, :], **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                    out = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = out.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # decode
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, prompt_embeds.dtype)
        else:
            image, has_nsfw_concept = latents, None

        do_denorm = [True] * image.shape[0] if has_nsfw_concept is None else [not x for x in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denorm)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
