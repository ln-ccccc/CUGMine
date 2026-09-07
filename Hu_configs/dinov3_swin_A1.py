crop_size = (512, 512)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[42.72, 46.32, 40.33],
    size=(512, 512),
    std=[47.11, 47.57, 48.98],
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=128),
    type='SegDataPreProcessor'
)
data_root = '/home/featurize/data/yunnan_dataset'
dataset_type = 'landsDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,  # 按epoch保存 checkpoint
        interval=1,     # 每1个epoch保存一次
        max_keep_ckpts=3,
        save_best='mIoU',
        type='CheckpointHook'
    ),
    logger=dict(interval=100, log_metric_by_epoch=True, type='LoggerHook'),  # 按epoch记录日志
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook')
)
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)
img_norm_cfg = dict(
    mean=[42.72, 46.32, 40.33],
    std=[47.11, 47.57, 48.98],
    to_rgb=False
)
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
img_scale = (512, 512)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)  # 按epoch处理日志

# 续训配置（按epoch续训）
resume_from = None # 'work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_upernet_swinmainV3/best_mIoU_iter_50500.pth'
resume = False  # 启用续训

model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=512,
        in_index=2,
        loss_decode=[
                    dict(type='LovaszLoss',loss_weight=1.0, per_image=True)],
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) ,
        num_classes=6,
        out_channels=6,
        num_convs=1,
        type='FCNHead'
    ),
    backbone=dict(
        type='DINOv3SwinEncoder',
        dinov3_out_channels=[1024, 1024, 1024, 1024],
        dinov3_cfg=dict(
            adapt_patch_size='center_padding',
            frozen_stages=-1,
            img_size=512,
            model_name='dinov3_vitl16',
            output_cls_token=False,
            repo_dir='/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/dinov3',
            weights='/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
        ),
        swin_cfg=dict(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            out_indices=(0, 1, 2, 3),
            patch_size=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            norm_cfg=dict(type='LN'),
            patch_norm=True,
            act_cfg=dict(type='GELU'),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/swin_base_patch4_window7_224_20220317-e9b98025.pth'
            ),
        ),
        feature_adapt_cfg=None,
        fusion_cfg=dict(
            type='cross_stage_progressive',
            act_cfg=dict(type='ReLU'),
            fusion_order='low2high',
            in_channels_list=[128, 256, 512, 1024],
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) ,
            intra_stage_type='attention',
            attn_channels=384
        )
    ),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[42.72, 46.32, 40.33],
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512),
        std=[47.11, 47.57, 48.98],
        test_cfg=dict(size_divisor=128),
        type='SegDataPreProcessor'
    ),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        loss_decode=[
                    dict(type='LovaszLoss',loss_weight=1.0, per_image=True)],
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) ,
        num_classes=6,
        out_channels=6,
        pool_scales=(1, 2, 3, 6),
        type='UPerHead'
    ),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder'
)

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) 
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(0.9, 0.999),
        eps=1e-08,
        lr=8e-5,  # 按epoch训练时学习率通常与iter保持一致，通过总轮次调整
        type='AdamW',
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=1.0, decay_mult=1.0),
            'backbone.feature_adapt_layers': dict(lr_mult=1.0, decay_mult=1.0),
            'norm': dict(decay_mult=0.0)
        },
    ),
    type='OptimWrapper'
)
# 按epoch的学习率调度器（总轮次30，前5轮线性热身，之后余弦退火）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,  # 前5个epoch热身
        convert_to_iter_based=True  # 转换为基于迭代的调度（内部处理）
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=25,  # 余弦周期（30-5=25）
        eta_min=8e-6,
        by_epoch=True,
        begin=5,
        end=30,
        convert_to_iter_based=True
    )
]
randomness = dict(seed=0)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/featurize/data/yunnan_dataset',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ],
        type='landsDataset'
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
test_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    type='IoUMetric'
)

# 训练配置改为按epoch（总30轮，每1轮验证一次）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
train_dataloader = dict(
        batch_size=2,
        dataset=dict(
            data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
            data_root='/home/featurize/data/yunnan_dataset',
            pipeline=[
                dict(type='LoadSingleRSImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
                dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs'),
            ],
            type='landsDataset'
        ),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=True, type='DefaultSampler')  # 按epoch训练用DefaultSampler
    )
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadSingleRSImageFromFile'),
    dict(
        transforms=[
            [dict(keep_ratio=True, scale_factor=1.0, type='Resize')],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ],
        type='TestTimeAug'
    )
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/featurize/data/yunnan_dataset',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs'),
        ],
        type='landsDataset'
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
val_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    type='IoUMetric'
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)
cudnn_benchmark = True
work_dir = '/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/work_dirs_A1/dinov3_swin_upernet_swinmainE_A1'
