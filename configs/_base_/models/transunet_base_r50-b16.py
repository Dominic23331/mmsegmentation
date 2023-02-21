norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255, 255, 255],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TransUnet',
        img_size=(224, 224),
        in_channels=3,
        hidden_size=768,
        num_layers=12,
        mlp_dim=3072,
        num_heads=12,
        resnet_layers=(3, 4, 9),
        resnet_width_factor=1,
        head_channels=512,
        decoder_channels=(256, 128, 64, 16),
        n_skip=3,
        skip_channels=[512, 256, 64, 16],
        dropout_rate=0.1,
        attention_dropout_rate=0.,
        grid=(14, 14)),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        in_index=4,
        channels=16,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.5),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
