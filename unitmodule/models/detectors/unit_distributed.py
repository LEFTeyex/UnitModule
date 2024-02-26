from typing import Dict, Union

import torch
from mmengine.model.utils import detect_anomalous_params
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS


def ddp_train_step_with_unit_module(self, data: Union[dict, tuple, list],
                                    optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    with optim_wrapper.optim_context(self):
        data, unit_losses = self.module.data_preprocessor(data, training=True)
        losses = self._run_forward(data, mode='loss')
    losses.update(unit_losses)
    parsed_loss, log_vars = self.module.parse_losses(losses)
    optim_wrapper.update_params(parsed_loss)
    if self.detect_anomalous_params:
        detect_anomalous_params(parsed_loss, model=self)
    return log_vars


# switch MMDistributedDataParallel train_step and register it
MMDistributedDataParallel.train_step = ddp_train_step_with_unit_module
MODEL_WRAPPERS.register_module(module=MMDistributedDataParallel, force=True)
