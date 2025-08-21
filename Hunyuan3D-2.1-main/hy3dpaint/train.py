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

import os, sys
import argparse
import subprocess
import warnings

import torch
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from src.utils.train_util import instantiate_from_config

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-r", "--resume", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--resume_weights_only", action="store_true", help="only resume model weights")
    parser.add_argument("-b", "--base", type=str, default="base_config.yaml", help="path to base configs")
    parser.add_argument("-n", "--name", type=str, default="", help="experiment name")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes to use")
    parser.add_argument("--gpus", type=str, default="0,", help="gpu ids to use")
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging data")
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "project.yaml"))


class CodeSnapshot(Callback):
    """Minimal code snapshotter (disabled copying by default)."""

    def __init__(self, savedir):
        self.savedir = savedir

    def get_file_list(self):
        return [
            b.decode()
            for b in set(subprocess.check_output('git ls-files -- ":!:configs/*"', shell=True).splitlines())
            | set(subprocess.check_output("git ls-files --others --exclude-standard", shell=True).splitlines())
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        # If you want to copy files, uncomment below
        # for f in self.get_file_list():
        #     if not os.path.exists(f) or os.path.isdir(f):
        #         continue
        #     os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
        #     shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except Exception:
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


# -------- helpers --------

def get_unet_core(pipeline_or_module):
    """Return inner UNet whether wrapped or not, without creating self-references."""
    unet = getattr(pipeline_or_module, "unet", pipeline_or_module)
    return getattr(unet, "unet", unet)


def has_self_reference(mod) -> bool:
    try:
        return hasattr(mod, "unet") and (mod.unet is mod)
    except Exception:
        return False


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    torch.set_float32_matmul_precision("medium")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name + exp_name)

    # auto-resume
    if os.path.exists(logdir) and opt.resume is None:
        auto_resume_path = os.path.join(logdir, "checkpoints", "last.ckpt")
        if os.path.exists(auto_resume_path):
            opt.resume = auto_resume_path
            print(f"Auto set resume ckpt {opt.resume}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")

    node_rank = int(os.environ.get("NODE_RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_gpus_per_node = torch.cuda.device_count()

    global_rank = node_rank * num_gpus_per_node + local_rank
    seed_everything(opt.seed + global_rank)

    # load configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer

    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    try:
        ngpu = int(opt.gpus)
    except Exception:
        ngpu = len(opt.gpus.strip(",").split(","))
    trainer_config["devices"] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # build model from config
    model = instantiate_from_config(config.model)

    # ---- robust UNet access (no self-loop) ----
    # never set model.unet.unet = model.unet
    if has_self_reference(model.unet):
        raise RuntimeError("Detected self-referencing UNet (model.unet.unet is model.unet). Remove that alias.")

    model_unet = get_unet_core(model.unet)
    model_unet_prefix = "unet.unet." if hasattr(model.unet, "unet") else "unet."

    # init from checkpoints (optional)
    if getattr(config, "init_unet_from", None):
        unet_ckpt_path = config.init_unet_from
        sd = torch.load(unet_ckpt_path, map_location="cpu")
        model_unet.load_state_dict(sd, strict=True)

    if getattr(config, "init_vae_from", None):
        vae_ckpt_path = config.init_vae_from
        sd_vae = torch.load(vae_ckpt_path, map_location="cpu")

        def replace_key(key_str):
            replace_pairs = [("key", "to_k"), ("query", "to_q"), ("value", "to_v"), ("proj_attn", "to_out.0")]
            for a, b in replace_pairs:
                key_str = key_str.replace(a, b)
            return key_str

        sd_vae = {replace_key(k): v for k, v in sd_vae.items()}
        model.pipeline.vae.load_state_dict(sd_vae, strict=True)

    # optional controlnet
    if hasattr(model.unet, "controlnet"):
        if getattr(config, "init_control_from", None):
            unet_ckpt_path = config.init_control_from
            sd_control = torch.load(unet_ckpt_path, map_location="cpu")
            model.unet.controlnet.load(sd_control, strict=True)

    # adjust noise in-channels if requested
    noise_in_channels = config.model.params.get("noise_in_channels", None)
    if noise_in_channels is not None:
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                noise_in_channels,
                model_unet.conv_in.out_channels,
                model_unet.conv_in.kernel_size,
                model_unet.conv_in.stride,
                model_unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, : model_unet.conv_in.in_channels, :, :].copy_(model_unet.conv_in.weight)
            new_conv_in.bias.zero_()
            new_conv_in.bias[: model_unet.conv_in.bias.size(0)].copy_(model_unet.conv_in.bias)
            model_unet.conv_in = new_conv_in

    # adjust controlnet in-channels
    if hasattr(model.unet, "controlnet"):
        if config.model.params.get("control_in_channels", None):
            control_in_channels = config.model.params.control_in_channels
            model.unet.controlnet.config["conditioning_channels"] = control_in_channels
            condition_conv_in = model.unet.controlnet.controlnet_cond_embedding.conv_in

            new_condition_conv_in = torch.nn.Conv2d(
                control_in_channels,
                condition_conv_in.out_channels,
                kernel_size=condition_conv_in.kernel_size,
                stride=condition_conv_in.stride,
                padding=condition_conv_in.padding,
            )

            with torch.no_grad():
                new_condition_conv_in.weight[:, : condition_conv_in.in_channels, :, :] = condition_conv_in.weight
                if condition_conv_in.bias is not None:
                    new_condition_conv_in.bias = condition_conv_in.bias

            model.unet.controlnet.controlnet_cond_embedding.conv_in = new_condition_conv_in

    rank_zero_print(f"Loaded Init ...")

    # resume partial weights
    if getattr(config, "resume_from", None):
        cnet_ckpt_path = config.resume_from
        sds = torch.load(cnet_ckpt_path, map_location="cpu")["state_dict"]
        sd0 = {k[len(model_unet_prefix) :]: v for k, v in sds.items() if model_unet_prefix in k}
        model_unet.load_state_dict(sd0, strict=True)
        if hasattr(model.unet, "controlnet"):
            sd1 = {k[16:]: v for k, v in sds.items() if "unet.controlnet." in k}
            model.unet.controlnet.load_state_dict(sd1, strict=True)
        rank_zero_print(f"Loaded {cnet_ckpt_path} ...")

    # resume full module
    if opt.resume and opt.resume_weights_only:
        model = model.__class__.load_from_checkpoint(opt.resume, **config.model.params)

    model.logdir = logdir

    # ----- trainer & callbacks -----
    trainer_kwargs = {}

    # logger
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {"name": "tensorboard", "save_dir": logdir, "version": "0"},
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # checkpoint callback
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,
        },
    }

    modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step"},
        },
        "code_snapshot": {"target": "train.CodeSnapshot", "params": {"savedir": codedir}},
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, lightning_config.get("callbacks", OmegaConf.create()))
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # precision & strategy (optional override by CLI/PL)
    trainer_kwargs["precision"] = "bf16"
    trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)

    # trainer from config (avoid double-assigning strategy/accelerator)
    cfg = config["lightning"]["trainer"]
    trainer = Trainer(**cfg, inference_mode=False)
    trainer.logdir = logdir

    # ----- data -----
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    # ----- LR setup -----
    base_lr = config.model.base_learning_rate
    accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # ----- fit -----
    if opt.resume and not opt.resume_weights_only:
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:

        trainer.fit(model, data)
