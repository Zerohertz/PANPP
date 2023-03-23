model = dict(
    type='PAN_PP',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(256, 512, 1024, 2048),
        out_channels=128,
        fpems=2
    ),
    detection_head=dict(
        type='PAN_PP_DetHead',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False,
    )
)
data = './data/TestData/image/1546387085517921.jpg'
test_cfg = dict(
    min_score=0.75,
    min_area=260,
    min_kernel_area=0.1,
    scale=2,
    bbox_type='rect',
    result_path='outputs',
    pretrain='./pretrained/doc_panpp_best_weight.pth.tar'
)