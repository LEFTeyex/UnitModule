from typing import Optional, Tuple, Union

import mmcv.cnn as cnn
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor


class LargeKernelLayer(BaseModule):
    def __init__(self,
                 channels: int,
                 large_kernel: int,
                 small_kernel: int,
                 padding_mode: str = 'reflect',
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        common_kwargs = dict(padding_mode=padding_mode,
                             groups=channels,
                             norm_cfg=norm_cfg,
                             act_cfg=None)

        self.dw_large = cnn.ConvModule(channels, channels, large_kernel,
                                       padding=large_kernel // 2, **common_kwargs)
        self.dw_small = cnn.ConvModule(channels, channels, small_kernel,
                                       padding=small_kernel // 2, **common_kwargs)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x) -> Tensor:
        x_large = self.dw_large(x)
        x_small = self.dw_small(x)
        return self.act(x_large + x_small)


class LKBlock(BaseModule):
    def __init__(self,
                 channels: int,
                 large_kernel: int,
                 small_kernel: int,
                 dw_ratio: float = 1.0,
                 padding_mode: str = 'reflect',
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        dw_channels = int(channels * dw_ratio)

        self.pw1 = cnn.ConvModule(channels, dw_channels, 1, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dw = LargeKernelLayer(dw_channels, large_kernel, small_kernel,
                                   padding_mode=padding_mode,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.pw2 = cnn.ConvModule(dw_channels, channels, 1, 1,
                                  norm_cfg=norm_cfg, act_cfg=None)
        self.norm = build_norm_layer(norm_cfg, channels)[1]

    def forward(self, x) -> Tensor:
        y = self.pw1(x)
        y = self.dw(y)
        y = self.pw2(y)
        x = self.norm(x + y)
        return x


@MODELS.register_module()
class UnitBackbone(BaseModule):
    def __init__(self,
                 stem_channels: Tuple[int],
                 large_kernels: Tuple[int],
                 small_kernels: Tuple[int],
                 in_channels: int = 3,
                 dw_ratio: float = 1.0,
                 padding_mode: str = 'reflect',
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        assert len(large_kernels) == len(small_kernels)
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        inc = in_channels

        stem_layers = []
        for outc in stem_channels:
            stem_layers.append(
                cnn.ConvModule(inc, outc, 3, 2,
                               padding=1, padding_mode=padding_mode,
                               norm_cfg=norm_cfg, act_cfg=act_cfg))
            inc = outc
        self.stem = nn.Sequential(*stem_layers)

        layers = []
        for large_k, small_k in zip(large_kernels, small_kernels):
            layers.append(
                LKBlock(inc, large_k, small_k, dw_ratio,
                        padding_mode, norm_cfg, act_cfg))
        self.layers = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return x


@MODELS.register_module()
class THead(BaseModule):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 out_channels: int = 3,
                 padding_mode: str = 'reflect',
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = cnn.ConvModule(in_channels, hid_channels, 3, 1,
                                    padding=1, padding_mode=padding_mode,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = cnn.ConvModule(hid_channels, out_channels, 3, 1,
                                    padding=1, padding_mode=padding_mode,
                                    norm_cfg=None, act_cfg=None)

    def forward(self, x) -> Tensor:
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = torch.sigmoid(x)
        return x


@MODELS.register_module()
class AHead(BaseModule):
    def __init__(self,
                 mean_dim: Union[int, Tuple[int]] = (-2, -1),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.mean_dim = mean_dim

    def forward(self, x) -> Tensor:
        return torch.mean(x, dim=self.mean_dim, keepdim=True)


@MODELS.register_module()
class UnitModule(BaseModule):
    def __init__(self,
                 unit_backbone: dict,
                 t_head: dict,
                 a_head: dict,
                 loss_t: dict,
                 loss_acc: Optional[dict] = None,
                 loss_cc: Optional[dict] = None,
                 loss_sp: Optional[dict] = None,
                 loss_tv: Optional[dict] = None,
                 alpha: float = 0.9,
                 t_min: float = 0.001,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert 0 < alpha < 1
        assert 0 <= t_min < 0.1

        self.alpha = alpha
        self.t_min = t_min

        self.unit_backbone = MODELS.build(unit_backbone)
        self.t_head = MODELS.build(t_head)
        self.a_head = MODELS.build(a_head)

        self.loss_t = MODELS.build(loss_t)
        self.loss_acc = MODELS.build(loss_acc) if loss_acc else None
        self.loss_cc = MODELS.build(loss_cc) if loss_cc else None
        self.loss_sp = MODELS.build(loss_sp) if loss_sp else None
        self.loss_tv = MODELS.build(loss_tv) if loss_tv else None

    def forward(self, x, training: bool = False) -> Union[Tensor, Tuple[Tensor, dict]]:
        if training:
            return self.loss(x)
        else:  # training == False
            return self.predict(x)

    def _forward(self, x) -> Tuple[Tensor, Tensor]:
        feature = self.unit_backbone(x)
        t = self.t_head(feature)
        a = self.a_head(x)
        return t, a

    def predict(self, x, show: bool = False) -> Union[Tensor, tuple]:
        t, a = self._forward(x)
        t = torch.clamp(t, min=self.t_min)

        x = self.denoise(x, t, a)
        x = torch.clamp(x, 0, 1)
        return (x, t, a) if show else x

    def loss(self, x) -> Tuple[Tensor, dict]:
        feature = self.unit_backbone(x)
        t = self.t_head(feature)
        a = self.a_head(x)

        t = torch.clamp(t, min=self.t_min)

        # get x of denoise
        x_denoise = self.denoise(x, t, a)

        # create fake x with noise and predict its t and A
        x_fake = self.noise(x, self.alpha, a)
        t_fake, a_fake = self._forward(x_fake)
        x_fake_denoise = self.denoise(x_fake, t_fake, a_fake)

        loss_t = self.loss_t(self.alpha * t, t_fake)
        losses = dict(loss_t=loss_t)
        if self.loss_acc:
            loss_acc = self.loss_acc(feature, a)
            losses.update(loss_acc=loss_acc)

        if self.loss_cc:
            loss_cc = self.loss_cc(x_denoise)
            losses.update(loss_cc=loss_cc)

        if self.loss_sp:
            loss_sp = self.loss_sp(x_denoise, x_fake_denoise)
            losses.update(loss_sp=loss_sp)

        if self.loss_tv:
            loss_tv = self.loss_tv(x_denoise)
            losses.update(loss_tv=loss_tv)

        x_denoise = torch.clamp(x_denoise, 0, 1)
        return x_denoise, losses

    @staticmethod
    def noise(x, t, a) -> Tensor:
        """Noise image"""
        return x * t + (1 - t) * a

    @staticmethod
    def denoise(x, t, a) -> Tensor:
        """Denoise image"""
        return (x - (1 - t) * a) / t
