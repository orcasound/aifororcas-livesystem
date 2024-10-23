"""
Module: L2Loss.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import torch.nn as nn

""" L2 (MSE) Loss  """
class L2Loss(nn.Module):

    def __init__(self, freq_dim=-1, reduction="none"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.freq_dim = freq_dim
        self.reduction = reduction

    def __call__(self, ground_truth, prediction):
        mse_ = self.mse(ground_truth, prediction).sum()
        return mse_