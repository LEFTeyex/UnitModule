import copy
from functools import reduce
from numbers import Number
from typing import Sequence, List, Tuple, Optional, Union

import numpy as np
import torch.nn.functional as F
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmyolo.models.data_preprocessors import YOLOv5DetDataPreprocessor


def sum_dict(a, b):
    temp = dict()
    for key in (a.keys() | b.keys()):
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


@MODELS.register_module()
class UnitDetDataPreprocessor(DetDataPreprocessor, BaseModule):
    def __init__(self,
                 unit_module: dict,
                 pad_mode: str = 'reflect',
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None,
                 init_cfg=None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)

        # BaseModule __init__
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        assert pad_mode in ('reflect', 'circular'), \
            f"Excepted ('reflect', 'circular'), but got {pad_mode}"
        self.pad_mode = pad_mode
        self.unit_module = MODELS.build(unit_module)

    def forward(self,
                data: dict,
                training: bool = False) -> Union[Tuple[dict, dict], dict]:
        data = self.cast_data(data)
        data['inputs'], losses = self.unit_module_forward(data['inputs'], training)

        data = super(UnitDetDataPreprocessor, self).forward(data, training)
        return (data, losses) if training else data

    def unit_module_forward(self, batch_inputs, training: bool = False) -> Tuple[list, dict]:
        outputs = []
        losses = []
        for batch_input in batch_inputs:
            # padding
            oh, ow = batch_input.shape[1:]
            pad_h = int(np.ceil(oh / self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(ow / self.pad_size_divisor)) * self.pad_size_divisor
            p2d = (0, (pad_w - ow), 0, (pad_h - oh))
            batch_input = batch_input.float()
            batch_input_pad = F.pad(batch_input, p2d, self.pad_mode)

            # UnitModule forward
            batch_input_pad = batch_input_pad.unsqueeze(0) / 255.
            if training:
                batch_output_pad, _losses = self.unit_module(batch_input_pad, training)
                losses.append(_losses)
            else:
                batch_output_pad = self.unit_module(batch_input_pad, training)
            batch_output_pad = batch_output_pad.squeeze(0)

            # remove padding
            batch_output = batch_output_pad[..., :oh, :ow] * 255.
            outputs.append(batch_output)

        if training:
            n = len(losses)
            losses = reduce(sum_dict, losses)
            for k, v in losses.items():
                losses[k] = v / n

        return outputs, losses


@MODELS.register_module()
class UnitYOLOv5DetDataPreprocessor(YOLOv5DetDataPreprocessor, BaseModule):
    def __init__(self,
                 unit_module: dict,
                 pad_mode: str = 'reflect',
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = True,
                 batch_augments: Optional[List[dict]] = None,
                 init_cfg=None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)

        # BaseModule __init__
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        assert pad_mode in ('reflect', 'circular'), \
            f"Excepted ('reflect', 'circular'), but got {pad_mode}"
        self.pad_mode = pad_mode
        self.unit_module = MODELS.build(unit_module)

    def forward(self,
                data: dict,
                training: bool = False) -> Union[Tuple[dict, dict], dict]:
        data = self.cast_data(data)
        data['inputs'], losses = self.unit_module_forward(data['inputs'], training)

        data = super(UnitYOLOv5DetDataPreprocessor, self).forward(data, training)
        return (data, losses) if training else data

    def unit_module_forward(self, batch_inputs, training: bool = False) -> Tuple[list, dict]:
        losses = {}
        if training:
            batch_inputs = batch_inputs.float()
            batch_inputs = batch_inputs / 255.
            batch_inputs, losses = self.unit_module(batch_inputs, training)
            outputs = batch_inputs * 255.
        else:
            outputs = []
            for batch_input in batch_inputs:
                # padding
                oh, ow = batch_input.shape[1:]
                pad_h = int(np.ceil(oh / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(ow / self.pad_size_divisor)) * self.pad_size_divisor
                p2d = (0, (pad_w - ow), 0, (pad_h - oh))
                batch_input = batch_input.float()
                batch_input_pad = F.pad(batch_input, p2d, self.pad_mode)

                # UnitModule forward
                batch_input_pad = batch_input_pad.unsqueeze(0) / 255.
                batch_output_pad = self.unit_module(batch_input_pad, training)
                batch_output_pad = batch_output_pad.squeeze(0)

                # remove padding
                batch_output = batch_output_pad[..., :oh, :ow] * 255.
                outputs.append(batch_output)

        return outputs, losses
