import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor
from torchvision.ops import RoIPool


@MODELS.register_module()
class AssistingColorCastLoss(nn.Module):
    def __init__(self, channels: int, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.roi_pooling = RoIPool((7, 7), 1)
        self.down_conv = nn.Conv2d(channels, 3, 1, 1)
        self.acc_head = nn.Sequential(
            nn.Linear(49, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1))

    def forward(self, feature: Tensor, a: Tensor) -> Tensor:
        device = feature.device
        b, _, h, w = feature.shape
        a = a.squeeze(-1).squeeze(-1)  # (b, 3)
        boxes = [torch.tensor(
            [[0, 0, h - 1, w - 1]],
            dtype=torch.float32).to(device) for _ in range(b)]

        feature = self.roi_pooling(feature, boxes)
        feature = self.down_conv(feature).view(b, 3, -1)
        color_cast = self.acc_head(feature).squeeze(-1)  # (b, 3)

        loss = self.loss_fn(color_cast, a)
        return self.loss_weight * loss
