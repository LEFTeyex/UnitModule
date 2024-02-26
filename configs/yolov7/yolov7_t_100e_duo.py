_base_ = [
    '../_base_/datasets/duo_detection_mmyolo.py',
    '../_base_/default_runtime_mmyolo.py',
]
env_cfg = dict(cudnn_benchmark=True)

max_epochs = 100
num_last_epochs = 15
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=10, dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=_base_.train_bs),
    constructor='YOLOv7OptimWrapperConstructor')
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs),
)
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

num_classes = 4
# anchors for DUO
anchors = [[(13, 12), (20, 18), (27, 25)],
           [(35, 31), (44, 39), (55, 52)],
           [(80, 45), (74, 69), (116, 102)]]
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv7Backbone',
        arch='Tiny',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='LeakyReLU', inplace=True, negative_slope=0.1)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(type='TinyDownSampleBlock', middle_ratio=0.25),
        upsample_feats_cat_first=False,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='LeakyReLU', inplace=True, negative_slope=0.1),
        is_tiny_version=True,
        use_repconv_outs=False),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.4],
        simota_candidate_topk=10,
        simota_iou_weight=3.0,
        simota_cls_weight=1.0),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))
