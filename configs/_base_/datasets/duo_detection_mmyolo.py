data_root = 'data/DUO/'

train_img_file = 'images/train'
val_img_file = 'images/test'
train_ann_file = 'annotations/instances_train.json'
val_ann_file = 'annotations/instances_test.json'

mean_bgr = [85.603, 148.034, 64.697]
std_bgr = [32.28, 39.201, 26.55]
mean_rgb = [64.697, 148.034, 85.603]
std_rgb = [26.55, 39.201, 32.28]

classes = ('holothurian', 'echinus', 'scallop', 'starfish')

img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
evaluator_type = 'mmdet.CocoMetric'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad',
         pad_to_square=True,
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad',
         pad_to_square=True,
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

num_gpu = 2
train_bs = 4
val_bs = 1
auto_scale_lr = dict(enable=False, base_batch_size=train_bs * num_gpu)
train_dataloader = dict(
    batch_size=train_bs,
    num_workers=train_bs,
    persistent_workers=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_file),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=val_bs,
    num_workers=val_bs * 2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_file),
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type=evaluator_type,
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
