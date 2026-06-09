from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)


def _merged_weight_norm_state(module):
    """Return weight-norm parameters that reproduce the module's current weight."""

    for hook in module._forward_pre_hooks.values():
        if getattr(hook, "name", None) != "weight":
            continue

        weight = module.weight.detach()
        return {
            "weight_g": torch.norm_except_dim(weight, 2, hook.dim),
            "weight_v": weight,
        }
    return {}


def merged_peft_state_dict(module):
    """Return a state dict with PEFT adapter weights merged into base layers."""

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return module.state_dict()

    if not any(isinstance(submodule, BaseTunerLayer) for submodule in module.modules()):
        return module.state_dict()

    adapter_layers = [
        submodule for submodule in module.modules()
        if isinstance(submodule, BaseTunerLayer)
    ]
    try:
        for submodule in adapter_layers:
            submodule.merge(safe_merge=True)

        weight_norm_overrides = {}
        for name, submodule in module.named_modules():
            if not isinstance(submodule, BaseTunerLayer):
                continue
            base_layer = submodule.get_base_layer()
            prefix = f"{name}.base_layer." if name else "base_layer."
            for key, value in _merged_weight_norm_state(base_layer).items():
                weight_norm_overrides[prefix + key] = value

        state_dict = {}
        for key, value in module.state_dict().items():
            if ".lora_" in key:
                continue
            value = weight_norm_overrides.get(key, value)
            state_dict[key.replace(".base_layer.", ".")] = value.detach().cpu().clone()
        return state_dict
    finally:
        for submodule in reversed(adapter_layers):
            submodule.unmerge()
