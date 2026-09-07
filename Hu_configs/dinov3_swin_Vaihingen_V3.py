# ============================================================
# dinov3_swin_Vaihingen_V2_optimized.py
# 完整配置文件（已整合所有优化建议）
# ============================================================

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
data_root = '/home/featurize/data/vaihingen'
dataset_type = 'ISPRSDataset'

default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best='mIoU',
        type='CheckpointHook'
    ),
    logger=dict(interval=100, log_metric_by_epoch=True, type='LoggerHook'),
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
log_processor = dict(by_epoch=True)

# ==================== 模型配置 ====================
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=512,
        in_index=2,
        # 【优化】损失函数组合 + 类别权重
        loss_decode=[
            dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=0.4,
                class_weight=[1.0, 1.0, 1.2, 1.0, 2.0, 0.0]
            ),
            dict(type='LovaszLoss', loss_weight=0.3, per_image=True)
        ],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
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
        feature_adapt_cfg=dict(
            conv_type='Conv2d',
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='GELU'),
            adapt_strides=[4, 2, 1, 1]
        ),
        fusion_cfg=dict(
            type='cross_stage_progressive',
            act_cfg=dict(type='ReLU'),
            fusion_order='low2high',
            in_channels_list=[128, 256, 512, 1024],
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
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
        # 【优化】损失函数组合 + 类别权重
        loss_decode=[
            dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1.0,
                class_weight=[1.0, 1.0, 1.2, 1.0, 2.0, 0.0]
                # impervious_surface: 1.0
                # building: 1.0  
                # low_vegetation: 1.2 (易与tree混淆)
                # tree: 1.0
                # car: 2.0 (小目标，加大权重)
                # clutter: 0.0 (忽略)
            ),
            dict(type='LovaszLoss', loss_weight=0.75, per_image=True)
        ],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
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

# ==================== 优化器配置 ====================
optim_wrapper = dict(
    # 【优化】grad_clip 从 0.01 改到 1.0
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        betas=(0.9, 0.999),
        eps=1e-08,
        lr=8e-5,
        type='AdamW',
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            # 先用保守版，只设置 norm 层不衰减
            # 确认参数名后可以加回差异化学习率
            'norm': dict(decay_mult=0.0)
        },
    ),
    type='OptimWrapper'
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=8e-6,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True
    )
]
randomness = dict(seed=0)

# ==================== 测试配置 ====================
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
test_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    type='IoUMetric'
)

# ==================== 训练配置 ====================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

# 【优化】数据增强：修正顺序 + 收敛参数
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            # 1. 先 RandomResize
            dict(
                type='RandomResize', 
                scale=(512, 512), 
                ratio_range=(0.5, 2.0), 
                keep_ratio=True
            ),
            # 2. 再 RandomRotate（在 crop 之前，裁剪会去掉黑边）
            dict(
                type='RandomRotate', 
                prob=0.25,           # 保守概率
                degree=(-10, 10),    # 小角度更稳定
                pad_val=0,
                seg_pad_val=255
            ),
            # 3. 然后 RandomCrop
            dict(
                type='RandomCrop', 
                crop_size=crop_size, 
                cat_max_ratio=0.75
            ),
            # 4. RandomFlip
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            # 5. PhotoMetricDistortion（参数收敛，避免过度扰动）
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=16,         # 从32降到16
                contrast_range=(0.8, 1.2),   # 收窄范围
                saturation_range=(0.8, 1.2), # 收窄范围
                hue_delta=10                 # 从18降到10
            ),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

# ==================== TTA 配置 ====================
tta_model = dict(type='SegTTAModel')
# 【优化】多尺度 TTA + 移除 LoadAnnotations
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            # 多尺度
            [
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
            ],
            # 翻转
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            # 【修正】移除 LoadAnnotations，纯推理不需要
            [dict(type='PackSegInputs')],
        ],
        type='TestTimeAug'
    )
]

# ==================== 验证配置 ====================
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
val_evaluator = dict(
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    type='IoUMetric'
)

# ==================== 可视化配置 ====================
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)

cudnn_benchmark = True
work_dir = '/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_Vaihingen_V3'