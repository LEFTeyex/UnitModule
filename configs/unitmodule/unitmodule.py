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
    loss_sp=dict(type='SaturatedPixelLoss', loss_weight=0.01),
    loss_tv=dict(type='TotalVariationLoss', loss_weight=0.01),
    loss_cc=dict(type='ColorCastLoss', loss_weight=0.1),
    loss_acc=dict(type='AssistingColorCastLoss', channels=c_s2, loss_weight=0.1),
    alpha=0.9,
    t_min=0.001)
