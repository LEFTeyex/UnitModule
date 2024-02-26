from .unit_detectors import (UnitCascadeRCNN, UnitDETR, UnitDINO,
                             UnitFasterRCNN, UnitFCOS, UnitRetinaNet,
                             UnitTOOD, UnitYOLODetector)


def register_unit_distributed(cfg):
    if cfg.get('with_unit_module'):
        # switch MMDistributedDataParallel to fit model with UnitModule
        import unitmodule.models.detectors.unit_distributed


__all__ = [
    'UnitCascadeRCNN', 'UnitDETR', 'UnitDINO',
    'UnitFasterRCNN', 'UnitFCOS', 'UnitRetinaNet',
    'UnitTOOD', 'UnitYOLODetector',
    'register_unit_distributed',
]
