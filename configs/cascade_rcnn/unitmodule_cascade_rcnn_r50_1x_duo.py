_base_ = [
    './cascade_rcnn_r50_1x_duo.py',
    '../unitmodule/unitmodule.py',
]

model = dict(
    type='UnitCascadeRCNN',
    data_preprocessor=dict(
        type='UnitDetDataPreprocessor',
        unit_module=_base_.unit_module)
)

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(type='UnderwaterColorRandomTransfer', hue_delta=5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
