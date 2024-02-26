import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class TotalVariationLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w, = x.shape
        h_tv = self.loss_fn(x[:, :, 1:, :], x[:, :, :h - 1, :])
        w_tv = self.loss_fn(x[:, :, :, 1:], x[:, :, :, :w - 1])
        loss = h_tv + w_tv
        return self.loss_weight * loss
