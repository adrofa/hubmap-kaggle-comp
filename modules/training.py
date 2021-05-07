from modules.data import HubmapDataset

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
import sys
from tqdm import tqdm
import random
import gc

gen_tiles = HubmapDataset.gen_tiles


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def logit(x):
    return -1 * np.log(1 / x - 1)


class TrainDataset(Dataset):
    """Dataset for training. Returns:
        (1) augmented tile (part of an image);
        (2) augmented mask (target).
    """
    def __init__(self, db_zarr, tile_df, transform=None):
        """
        Args:
            db_zarr (zarr.Group): Zarr-group with images and its masks (target), where
                image is accessible by db_zarr["<image_id>"]["img"]
                mask is accessible by db_zarr["<image_id>"]["target"]
            tile_df (pandas.DataFrame): dataframe, which consists of 2 columns:
                "id" - image_id;
                "tile" - tuple with tile coordinates (x1, x2, y1, y2).
            transform (albumentations): augmentation.
        """
        self.zarr_db = db_zarr
        self.tiles = [row for _, row in tile_df.iterrows()]
        if transform:
            self.transform = A.Compose([transform, ToTensorV2(p=1)])
        else:
            self.transform = ToTensorV2(p=1)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        id_ = self.tiles[idx]["id"]
        coords = self.tiles[idx]["tile"]
        slc = np.s_[coords[0]: coords[1], coords[2]: coords[3]]
        img = self.zarr_db[id_]["img"][slc]
        tgt = self.zarr_db[id_]["target"][slc]
        transformed = self.transform(image=img, mask=tgt)
        img = transformed["image"].type(torch.float32)
        tgt = transformed["mask"].unsqueeze(0)
        return img, tgt


class ValidDataset(TrainDataset):
    """Dataset for validation. Returns:
        (1) augmented tile (part of an image);
        (2) augmented mask (target);
        (3) original mask (target).logit
    """
    def __getitem__(self, idx):
        id_ = self.tiles[idx]["id"]
        coords = self.tiles[idx]["tile"]
        slc = np.s_[coords[0]: coords[1], coords[2]: coords[3]]
        img = self.zarr_db[id_]["img"][slc]
        tgt = self.zarr_db[id_]["target"][slc]
        transformed = self.transform(image=img, mask=tgt)
        return (transformed["image"].type(torch.float32),
                transformed["mask"].unsqueeze(0),
                ToTensorV2(p=1)(image=tgt)["image"])


class InferenceDataset(Dataset):
    """Dataset for inference. Returns:
        (1) augmented tile (part of an image);
        (2) tile coordinates.
    """
    def __init__(self, img, tiles, transform=None):
        """
        Args:
            img (zarr.Group / numpy.array): image for prediction.
            tiles (list): list of tuples with tile coordinates [(x1, x2, y1, y2), ...].
        """
        self.img = img
        self.tiles = tiles
        if transform:
            self.transform = A.Compose([transform, ToTensorV2(p=1)])
        else:
            self.transform = ToTensorV2(p=1)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        coords = self.tiles[idx]
        img = self.img[coords[0]: coords[1], coords[2]: coords[3]]
        img = self.transform(image=img)["image"]
        return img.type(torch.float32), np.array(coords)


def train(model, dataloader, criterion, optimizer, device="cuda", verbose="train"):
    """Train model 1 epoch."""
    model.to(device)
    model.train()
    with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
        items_epoch = 0
        loss_epoch = 0
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            p = model.forward(x)
            loss = criterion(p, y.type_as(p))
            loss.backward()
            optimizer.step()

            items_batch = p.shape[0]
            loss_batch = loss.item() * items_batch

            items_epoch += items_batch
            loss_epoch += loss_batch

            loss = loss_epoch / items_epoch

            if verbose:
                log_batch = f"loss-batch: {loss_batch / items_batch:.5f}"
                log_epoch = f"loss-epoch: {loss:.5f}"
                iterator.set_postfix_str(" | ".join([log_batch, log_epoch]))
    return loss


def valid(model, dataloader, criterion, dice_ths=0.5, device="cuda", verbose="valid"):
    """Validate model. Computes:
        (1) loss (criterion);
        (2) dice score on the original image size.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        items_epoch = 0
        loss_epoch = 0
        intersection_epoch = 0
        true_epoch = 0
        pred_epoch = 0
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            for x, y, y_orig in iterator:
                x, y, y_orig = x.to(device), y.to(device), y_orig.to(device)
                p = model.forward(x)

                # loss
                items_epoch += p.shape[0]
                loss_epoch += criterion(p, y.type_as(p)).item() * p.shape[0]
                loss_runnig = loss_epoch / items_epoch

                if dice_ths:
                    # preds to binary mask of the original shape
                    p = F.interpolate(p, y_orig.shape[2:], mode="bilinear", align_corners=False)
                    p = torch.sigmoid(p)
                    p = (p > dice_ths).type_as(y_orig)

                    # dice
                    intersection_epoch += (p * y_orig).sum().item()
                    true_epoch += y_orig.sum().item()
                    pred_epoch += p.sum().item()
                    dice_running = (2 * intersection_epoch) / max((true_epoch + pred_epoch), 10 ** -5)
                else:
                    dice_running = "N/A"

                if verbose:
                    if dice_ths:
                        log_dice = f"dice: {dice_running:.5f}"
                    else:
                        log_dice = "dice: N/A"
                    log_loss = f"loss: {loss_runnig:.5f}"
                    iterator.set_postfix_str(" | ".join([log_dice, log_loss]))

    return dice_running, loss_runnig


def predict(storage, interpolate_to, dataloader, model,  device="cuda", verbose="predict"):
    """Writes (1) the sum of all logit predictions and (2) its total number into storage.
    Inasmuch image tiles may overlap there could be >1 prediction per pixel.

    Args:
        storage (zarr.Group): zarr-group with (1) 'pred' array and (2) 'count' array.
        interpolate_to (tuple):  shape of the original tile size.
        dataloader (torch.dataloader): should be built on InferenceDataset, which consists of
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            for x, coords in iterator:
                x, coords = x.to(device), coords.numpy()
                p = model.forward(x)
                p = F.interpolate(p, interpolate_to, mode="bilinear", align_corners=False)
                p = p.to("cpu")

                for i in range(len(p)):
                    p_i, c = p[i].numpy().squeeze(), coords[i]
                    slc = np.s_[c[0]: c[1], c[2]: c[3]]
                    storage["pred"][slc] = storage["pred"][slc] + p_i
                    storage["count"][slc] = storage["count"][slc] + 1


def pred2msk(msk, storage, dice_ths=0.5, chunk=1024, verbose="pred2msk"):
    """Converts sum of logit predictions into final binary prediction (1 or 0).

    Args:
        msk: array to store binary predictions.
        storage (zarr.Group): zarr-group with (1) 'pred' array and (2) 'count' array.
        dice_ths (float): threshold for binary prediction.
        chunk (int): size of array per operation (decrease in case of RAM limitations).
        verbose (str): logs description.
    """
    ths = logit(dice_ths)
    coords = gen_tiles(storage["pred"].shape, tile_size=(chunk, chunk), shift=(0, 0), last="do_nothing")
    with tqdm(coords, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
        for c in iterator:
            slc = np.s_[c[0]: c[1], c[2]: c[3]]
            pred = storage["pred"][slc]
            count = storage["count"][slc]

            msk[slc] = (pred / count) > ths

            del pred, count
            gc.collect()
