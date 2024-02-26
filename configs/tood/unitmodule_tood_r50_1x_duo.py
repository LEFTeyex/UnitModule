_base_ = [
    './tood_r50_1x_duo.py',
]

with_unit_module = True
norm_cfg = dict(type='GN', num_groups=8)
act_cfg = dict(type='ReLU')

k_1, k_2 = 9, 9
c_s1, c_s2 = 32, 32

unit_module = dict(
    type='UnitModule',
    unit_backbone=dict(
        type='UnitBackbone',
        stem_channels=(c_s1, c_s2),
        large_kernels=(k_1, k_2),
        small_kernels=(3, 3),
        dw_ratio=1.0,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    t_head=dict(
        type='THead',
        in_channels=c_s2,
        hid_channels=c_s2,
        out_channels=3,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    a_head=dict(type='AHead'),
    loss_t=dict(type='TransmissionLoss', loss_weight=500),
    loss_sp=dict(type='SaturatedPixelLoss', loss_weight=0.1),
    loss_tv=dict(type='TotalVariationLoss', loss_weight=0.01),
    loss_cc=dict(type='ColorCastLoss', loss_weight=0.1),
    loss_acc=dict(type='AssistingColorCastLoss', channels=c_s2, loss_weight=0.1),
    alpha=0.9,
    t_min=0.001)

model = dict(
    type='UnitTOOD',
    data_preprocessor=dict(
        type='UnitDetDataPreprocessor',
        unit_module=unit_module)
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
