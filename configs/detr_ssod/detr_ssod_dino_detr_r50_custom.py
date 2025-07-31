_base_ = "base_dino_detr_ssod_coco.py"

model = dict(
    bbox_head=dict(
        num_classes=3,
    )
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
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
    val=dict(
            ann_file="data/insulator-defect-detection/coco_ann/val/annotations/coco_instances.json",
            img_prefix="data/insulator-defect-detection/sly_project/val/img",
            classes=("broken", "insulator", "pollution-flashover"),
        ),
    test=dict(
            ann_file="data/insulator-defect-detection/coco_ann/test/annotations/coco_instances.json",
            img_prefix="data/insulator-defect-detection/sly_project/test/img",
            classes=("broken", "insulator", "pollution-flashover"),
        ),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 2],
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

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=50000)
evaluation = dict(type="SubModulesDistEvalHook", interval=1000)
lr_config = dict(step=[35000, 45000])
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=5, create_symlink=False)

work_dir = "output/split_250_1000/run_3"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
