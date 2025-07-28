_base_ = "base_dino_detr_ssod_coco.py"

model = dict(
    bbox_head=dict(
        num_classes=3,
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/insulator-defect-detection/ssl_split_250_1000/labeled.json",
            img_prefix="data/insulator-defect-detection/sly_project/train/img",
            classes=("broken", "insulator", "pollution-flashover"),

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/insulator-defect-detection/ssl_split_250_1000/unlabeled.json",
            img_prefix="data/insulator-defect-detection/sly_project/train/img",
            classes=("broken", "insulator", "pollution-flashover"),
        ),
    ),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
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
        unsup_weight=4.0,
        aug_query=False,
        
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=20000)
evaluation = dict(type="SubModulesDistEvalHook", interval=50)
lr_config = dict(step=[16000, 20000])
checkpoint_config = dict(by_epoch=False, interval=50, max_keep_ckpts=5, create_symlink=False)

work_dir = "output/split_250_1000"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
