"""
Module: summary.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import matplotlib
import numpy as np

matplotlib.use("Agg")

import torch

from PIL import Image, ImageDraw
from torchvision.utils import make_grid
from visualization.utils import spec2img

"""
Prepare given image data for tensorboard visualization
"""
def prepare_img(img, num_images=4, file_names=None):
    with torch.no_grad():
        if img.shape[0] == 0:
            raise ValueError("`img` must include at least 1 image.")
        img = img.unsqueeze(dim=0).transpose(0, 1)
        if num_images < img.shape[0]:
            tmp = img[:num_images]
        else:
            tmp = img
        tmp = spec2img(tmp)

        if file_names is not None:
            tmp = tmp.permute(0, 3, 2, 1)
            for i in range(tmp.shape[0]):
                try:
                    pil = Image.fromarray(tmp[i].numpy(), mode="RGB")
                    draw = ImageDraw.Draw(pil)
                    draw.text(
                        (2, 2),
                        os.path.basename(file_names[i]),
                        (255, 255, 255),
                    )
                    np_pil = np.asarray(pil).copy()
                    tmp[i] = torch.as_tensor(np_pil)
                except TypeError:
                    pass
            tmp = tmp.permute(0, 3, 1, 2)

        tmp = make_grid(tmp, nrow=1)
        return tmp.numpy()