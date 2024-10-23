"""
Module: metrics.py
Authors: Christian Bergler
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import torch

"""Divides two tensors element-wise, returning 0 if the denominator is <= 0."""
def _safe_div(numerator, denominator):
    t = torch.div(numerator, denominator)
    condition = torch.gt(denominator, float(0))
    return torch.where(condition, t, torch.zeros_like(t))

"""Defines a counter for for various metrics (e.g. confusion matrix values)."""
def _count_condition(condition, weights):
    with torch.no_grad():
        if weights is not None:
            if torch.is_tensor(weights):
                weights = weights.float()
            condition = torch.mul(condition.float(), weights)
        return condition.sum().item()

"""
Define of various evaluation metrics in order to track the entire network training, validation, and testing
"""
class MetricBase:
    def __init__(self):
        pass

    def reset(self, device=None):
        self.__init__(device=device)

    def update(self):
        pass

    def _get_tensor(self):
        pass

    def get(self):
        return self._get_tensor().item()

    def __str__(self):
        return str(self.get())

    def __format__(self, spec):
        return self.get().__format__(spec)

"""
Define sum metric
"""
class Sum(MetricBase):
    def __init__(self, device=None):
        self.value = torch.zeros(1, device=device)

    def update(self, values, weights=None):
        with torch.no_grad():
            if weights is not None:
                values = torch.mul(values, weights)
            self.value += values.sum()

    def _get_tensor(self):
        return self.value

"""
Define max metric
"""
class Max(MetricBase):
    def __init__(self, device=None):
        self.value = torch.zeros(1, dtype=torch.float, device=device)

    def update(self, values, weights=None):
        with torch.no_grad():
            if weights is not None:
                values = torch.mul(values, weights)

            tmp = values.max().float()
            if tmp > self.value:
                self.value = tmp

    def _get_tensor(self):
        return self.value

"""
Define mean metric
"""
class Mean(MetricBase):
    def __init__(self, device=None):
        self.total = torch.zeros(1, dtype=torch.float, device=device)
        self.count = torch.zeros(1, dtype=torch.float, device=device)

    def update(self, values, weights=None):
        with torch.no_grad():
            if weights is None:
                num_values = float(values.numel())
            else:
                if torch.is_tensor(weights):
                    num_values = weights.sum().float()
                    weights = weights.float()
                else:
                    num_values = torch.mul(values.numel(), weights).float()
                values = torch.mul(values, weights)

            self.total += values.sum().float()
            self.count += num_values

    def _get_tensor(self):
        return _safe_div(self.total, self.count).float()