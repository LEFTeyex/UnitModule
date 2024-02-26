_base_ = [
    './rtmdet_s_100e_duo.py',
    '../unitmodule/unitmodule.py',
]

model = dict(
    type='UnitYOLODetector',
    data_preprocessor=dict(
        type='UnitYOLOv5DetDataPreprocessor',
        unit_module=_base_.unit_module)
)

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(type='UnderwaterColorRandomTransfer', hue_delta=5),
    dict(type='mmdet.Pad',
         pad_to_square=True,
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
