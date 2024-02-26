import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class TransmissionLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        loss = self.loss_fn(a, b)
        return self.loss_weight * loss
