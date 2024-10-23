"""
Module: utils.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import torch

from .cm import apply_cm, viridis_cm

"""
Flip data along give dimension
Code from https://github.com/pytorch/pytorch/issues/229
Access Data: 06.02.2021, Last Access Date: 21.12.2021
Changes: Modified by Christian Bergler (06.02.2021)
"""
def flip(x, dim=-1):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]

"""
Converts a float spectrogram tensor to a uint8 image tensor.
"""
def spec2img(spec, normalize=True, cm=viridis_cm):
    with torch.no_grad():
        if spec.dim() == 4:
            dim = 1
            assert spec.size(1) == 1
        elif spec.dim() <= 3:
            dim = 0
            if spec.size(0) > 1 or spec.dim() == 2:
                spec = spec.unsqueeze(dim=0)
        else:
            raise ValueError("Unsupported spec dimension.")
        img = flip(spec, dim=-1)
        if normalize:
            img -= img.min()
            img /= img.max() + 1e-8
        img = img.mul(255).clamp(0, 255).long()
        img = apply_cm(img.cpu(), cm, dim=dim)
        return img.mul(255).clamp(0, 255).byte()
