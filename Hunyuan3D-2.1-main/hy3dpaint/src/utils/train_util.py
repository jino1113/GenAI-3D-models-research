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

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        m = importlib.import_module(module)
        importlib.reload(m)
    return getattr(importlib.import_module(module, package=None), cls)

def _to_py(o):
    if isinstance(o, (DictConfig, ListConfig)):
        return OmegaConf.to_container(o, resolve=True)
    return o

def instantiate_from_config(config):
    cfg = _to_py(config) or {}
    if not isinstance(cfg, dict) or "target" not in cfg:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(cfg["target"])
    params = _to_py(cfg.get("params", {})) or {}

    # ✅ ถ้ามีพารามิเตอร์ของ from_pretrained ให้เรียก from_pretrained แทน
    if any(k in params for k in ("pretrained_model_name_or_path", "model_name_or_path", "pretrained_model_path")) \
       and hasattr(cls, "from_pretrained"):
        return cls.from_pretrained(**params)

    # 🔁 รีเคอร์ซีฟอินสแตนซ์ซับคอนฟิก
    def resolve(v):
        v = _to_py(v)
        if isinstance(v, dict) and "target" in v:
            return instantiate_from_config(v)
        if isinstance(v, list):
            return [resolve(x) for x in v]
        return v

    params = {k: resolve(v) for k, v in params.items()}
    params.pop("kwargs", None)
    return cls(**params)
