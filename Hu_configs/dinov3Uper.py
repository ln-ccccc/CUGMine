crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        42.72,
        46.32,
        40.33,
    ],
    size=(
        512,
        512,
    ),
    std=[
        47.11,
        47.57,
        48.98,
    ],
    type='SegDataPreProcessor')
data_root = '/home/featurize/data/yunnan_dataset'
dataset_type = 'landsDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2500,
        max_keep_ckpts=1,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean=[
        42.72,
        46.32,
        40.33,
    ], std=[
        47.11,
        47.57,
        48.98,
    ], to_rgb=False)
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    512,
    512,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=1024,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(loss_weight=0.2, per_image=True, type='LovaszLoss'),
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        num_classes=6,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        adapt_patch_size='center_padding',
        frozen_stages=-1,
        img_size=512,
        model_name='dinov3_vitl16',
        output_cls_token=False,
        repo_dir=
        '/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/dinov3',
        type='DINOv3Backbone',
        weights=
        '/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    ),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            42.72,
            46.32,
            40.33,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            47.11,
            47.57,
            48.98,
        ],
        test_cfg=dict(size=(512,512)),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        dropout_ratio=0.1,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(type='LovaszLoss', loss_weight=0.4,reduction='none'),
            dict(type='DiceLoss', loss_weight=0.2)
        ],
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
        num_classes=6,
        pool_scales=(
            1,
            2,
            4,
            8,
        ),
       
        type='UPerHead'),
    pretrained=None,
    test_cfg=dict(mode='slide'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

optim_wrapper = dict(
    # 关键点1：显式声明类型
    type='OptimWrapper',  
    
    # 关键点2：优化器参数必须包裹在optimizer字段中
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-08
    ),
    
    # 参数分层配置（可选但推荐）
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'decode_head': dict(lr_mult=1.0),
            'auxiliary_head': dict(lr_mult=0.5)
        }
    ),
    
    # 梯度裁剪配置（可选）
    clip_grad=dict(max_norm=35, norm_type=2)
)


# 学习率调度器改进
param_scheduler = [
    dict(
        type='LinearLR',  # 增加1k迭代预热
        start_factor=1e-6,
        begin=0,
        end=1000,
        convert_to_iter_based=True),
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=1000,
        end=400000)
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/featurize/data/yunnan_dataset',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='landsDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=400000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='/home/featurize/data/yunnan_dataset',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='landsDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadSingleRSImageFromFile'),
    dict(
        transforms=[
            # [
            #     dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
            #     dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
            #     dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
            #     dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
            #     dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
            #     dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            # ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/featurize/data/yunnan_dataset',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='landsDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type=
             'LocalVisBackend'),
    ])
work_dir = '/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/Hu-dinov3swin'
