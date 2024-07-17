_base_ = [
    '../_base_/datasets/lsun-car_pad_512.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

# ada settings
aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}

model = dict(generator=dict(out_size=512), discriminator=dict(in_size=512))

data = dict(
    samples_per_gpu=4,
    train=dict(dataset=dict(imgs_root='./data/lsun/images/car')),
    val=dict(imgs_root='./data/lsun/images/car'))

ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])


total_iters = 1800002

metrics = dict(
    fid50k=dict(
        type='FID', 
        num_images=50000, 
        inception_pkl='work_dirs/inception_pkl/lsun-car-512-50k-rgb.pkl', 
        bgr2rgb=True),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000))

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID',
        num_images=50000,
        inception_pkl='work_dirs/inception_pkl/lsun-car-512-50k-rgb.pkl',
        bgr2rgb=True),
    sample_kwargs=dict(sample_model='ema'))
