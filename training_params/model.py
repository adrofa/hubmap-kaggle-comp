import segmentation_models_pytorch as smp


def get_model(version):

    if version == "unet_v1":
        return smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_depth=5,
            encoder_weights="imagenet",
        )

    elif version == "unet_v2":
        return smp.Unet(
            encoder_name="efficientnet-b2",
            encoder_depth=5,
            encoder_weights="imagenet",
        )

    elif version == "unet_v3":
        return smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_depth=5,
            encoder_weights="imagenet",
        )

    elif version == "unet_v4":
        return smp.Unet(
            encoder_name="efficientnet-b6",
            encoder_depth=5,
            encoder_weights="imagenet",
        )

    elif version == "unet_v5":
        return smp.Unet(
            encoder_name="resnet50",
            encoder_depth=5,
            encoder_weights="imagenet",
        )

    elif version == "unet_v6":
        return smp.Unet(
            encoder_name="xception",
            encoder_depth=5,
            encoder_weights="imagenet",
        )

    elif version == "fpn_v1":
        return smp.FPN(
            encoder_name='efficientnet-b2',
            encoder_depth=5,
            encoder_weights='imagenet'
        )

    else:
        raise Exception(f"Model version '{version}' is UNKNOWN!")
