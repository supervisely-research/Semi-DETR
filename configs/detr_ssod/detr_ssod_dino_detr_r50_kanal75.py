_base_ = "base_dino_detr_ssod_coco.py"

classes = ("horse", "horse head", "number plate", "rider", "yellow stick", "white stick")

model = dict(
    bbox_head=dict(
        num_classes=len(classes),
    )
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/kanal75/Training/coco_ann/train/annotations/coco_instances.json",
            img_prefix="data/kanal75/Training/train/img",
            classes=classes,

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/kanal75/frames_LOPP_2500.json",
            img_prefix="data/kanal75/frames_LOPP_2500",
            classes=classes,
        ),
    ),
    val=dict(
            ann_file="data/kanal75/Training/coco_ann/val/annotations/coco_instances.json",
            img_prefix="data/kanal75/Training/val/img",
            classes=classes,
        ),
    test=dict(
            ann_file="data/kanal75/Training/coco_ann/val/annotations/coco_instances.json",
            img_prefix="data/kanal75/Training/val/img",
            classes=classes,
        ),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 3],
            by_prob=True,
        )
    ),
)

semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=3.0,
        aug_query=False,
        
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=55000)
evaluation = dict(type="SubModulesDistEvalHook", interval=1000)
lr_config = dict(step=[40000, 50000])
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=5, create_symlink=False)

work_dir = "output/kanal75/split_400_2500/run_1"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
