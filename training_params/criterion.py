import torch
from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss
from torch.nn import BCEWithLogitsLoss


def get_criterion(version):
    if version == "focal_v1":
        return BinaryFocalLoss(alpha=0.04, gamma=2, reduction="mean",
                               ignore_index=None, normalized=False, reduced_threshold=None)

    elif version == "focal_v2":
        return BinaryFocalLoss(alpha=0.04, gamma=1.5, reduction="mean",
                               ignore_index=None, normalized=False, reduced_threshold=None)

    elif version == "bce_v1":
        return BCEWithLogitsLoss(pos_weight=torch.tensor(24.16), reduction='mean',
                                 weight=None, size_average=None, reduce=None)

    elif version == "bce_v2":
        return BCEWithLogitsLoss(reduction='mean',
                                 weight=None, size_average=None, reduce=None)

    elif version == "bce_v3":
        return BCEWithLogitsLoss(pos_weight=torch.tensor(15.463626008840938), reduction='mean',
                                 weight=None, size_average=None, reduce=None)

    elif version == "dice_v1":
        return DiceLoss(mode="binary", from_logits=True,
                        classes=None, log_loss=False, smooth=0.0, ignore_index=None, eps=1e-7)

    else:
        raise Exception(f"Criterion version '{version}' is UNKNOWN!")
