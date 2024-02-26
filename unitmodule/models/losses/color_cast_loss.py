import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class ColorCastLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x: Tensor) -> Tensor:
        x = torch.mean(x, dim=(-2, -1))
        # from color channel (0, 1, 2) corresponding to (1, 2, 0)
        loss = self.loss_fn(x, x[:, [1, 2, 0]])
        return self.loss_weight * loss
