import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class SaturatedPixelLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        zero = a.new_zeros(1)
        one = a.new_ones(1)

        loss_max = (torch.max(a, one) + torch.max(b, one) - 2 * one).nanmean()
        loss_min = -(torch.min(a, zero) + torch.min(b, zero)).nanmean()
        loss = loss_max + loss_min
        return self.loss_weight * loss
