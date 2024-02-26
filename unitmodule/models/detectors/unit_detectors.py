from typing import Dict, Union

import torch
from mmdet.models.detectors import (CascadeRCNN, DETR, DINO,
                                    FasterRCNN, FCOS, RetinaNet, TOOD)
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmyolo.models.detectors import YOLODetector


def train_step_with_unit_module(self, data: Union[dict, tuple, list],
                                optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    """With the UnitModule loss"""
    with optim_wrapper.optim_context(self):
        data, unit_losses = self.data_preprocessor(data, True)
        losses = self._run_forward(data, mode='loss')
    losses.update(unit_losses)
    parsed_losses, log_vars = self.parse_losses(losses)
    optim_wrapper.update_params(parsed_losses)
    return log_vars


def with_unit_module(cls):
    cls.train_step = train_step_with_unit_module
    return cls


@MODELS.register_module()
@with_unit_module
class UnitCascadeRCNN(CascadeRCNN):
    """CascadeRCNN with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitDETR(DETR):
    """DETR with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitDINO(DINO):
    """DINO with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitFasterRCNN(FasterRCNN):
    """FasterRCNN with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitFCOS(FCOS):
    """FCOS with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitRetinaNet(RetinaNet):
    """RetinaNet with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitTOOD(TOOD):
    """TOOD with UnitModule"""


@MODELS.register_module()
@with_unit_module
class UnitYOLODetector(YOLODetector):
    """YOLODetector with UnitModule"""
