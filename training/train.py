# Model training

from modules.data import HubmapDataset
from modules.training import TrainDataset, train, ValidDataset, valid

from training_params.model import get_model
from training_params.optimizer import get_optimizer
from training_params.criterion import get_criterion
from training_params.augmentation import get_augmentation
from training_params.scheduler import get_scheduler

import pandas as pd
import os
import json
import zarr
import torch
from torch.utils.data import DataLoader


CONFIG = {
    "version": "debug",

    "data_dir": r"data/",
    "out_dir": r"outputs/",
    "zarr_db_dir": r"outputs/zarr/v1/db.zarr",

    "model_version": "unet_v2",
    "model_weights": None,

    "optimizer_version": "adam_v1",
    "optimizer_weights": None,

    "scheduler_version": "rop_v1",

    "criterion_version": "dice_v1",
    "dice_ths": 0.5,

    "tiles_version": "v1",
    "augmentation_version": "v12",

    "batch_size": 8,

    "epoch_num": 1000,
    "early_stopping": 30,

    "debug": 3,
    "device": "cuda",
    "n_jobs": 2,
}


def main(cfg):
    dat = HubmapDataset(cfg["data_dir"], cfg["out_dir"])
    db = zarr.open_group(store=zarr.DirectoryStore(cfg["zarr_db_dir"]), mode="r")

    # results dir
    results_dir = dat.path.out / "models" / cfg["version"]
    if cfg["version"] == "debug":
        os.makedirs(results_dir, exist_ok=True)
    else:
        try:
            os.makedirs(results_dir, exist_ok=False)
        except:
            raise Exception(f"Version {cfg['version']} exists!")
    with open(results_dir / "config.json", "w") as f:
        json.dump(cfg, f, sort_keys=True, indent=4)

    # tiles
    tile_dct = dat.pkl_load(dat.path.out / "tiles" / cfg['tiles_version'] / "tile_dct.pkl")
    train_df = tile_dct["train_df"]
    valid_dct = tile_dct["valid_dct"]
    valid_id = list(valid_dct.keys())
    train_valid_df = tile_dct["train_valid_df"]
    # -- debug (cutting data)
    if cfg["debug"]:
        train_df = train_df.head(cfg["debug"])
        valid_dct = {id_: valid_dct[id_].head(cfg["debug"]) for id_ in valid_id}
        train_valid_df = train_valid_df.head(cfg["debug"])

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = {id_: ValidDataset(db, valid_dct[id_], augmentation["valid"]) for id_ in valid_id}
    dataset["train"] = TrainDataset(db, train_df, augmentation["train"])
    dataset["train_valid"] = ValidDataset(db, train_valid_df, augmentation["valid"])

    # PyTorch DataLoaders initialization
    dataloader = {
        id_: DataLoader(dataset[id_], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"])
        for id_ in valid_id
    }
    dataloader["train"] = DataLoader(
        dataset["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["n_jobs"])
    dataloader["train_valid"] = DataLoader(
        dataset["train_valid"], batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["n_jobs"])

    # model
    model = get_model(cfg["model_version"])
    if cfg["model_weights"]:
        model.load_state_dict(torch.load(cfg["model_weights"]))
    model.to(cfg["device"])

    # optimizer
    optimizer = get_optimizer(cfg["optimizer_version"], model.parameters())
    if cfg["optimizer_weights"]:
        optimizer.load_state_dict(torch.load(cfg["optimizer_weights"]))

    # scheduler
    if cfg["scheduler_version"]:
        scheduler = get_scheduler(cfg["scheduler_version"], optimizer)

    # loss function
    criterion = get_criterion(cfg["criterion_version"])

    # EPOCHS
    monitor = []
    loss_min = None
    dice_max = None
    epochs_without_improvement = 0
    for epoch in range(cfg["epoch_num"]):
        print(f"Epoch-{epoch}")
        monitor_epoch = {}

        # train
        train_loss_raw = train(model, dataloader["train"], criterion, optimizer,
                               device=cfg["device"], verbose="train-train")
        monitor_epoch["train_loss_raw"] = train_loss_raw
        # train loss
        _, train_loss = valid(model, dataloader["train_valid"], criterion, dice_ths=False,
                              device=cfg["device"], verbose="train-valid")
        monitor_epoch["train_loss"] = train_loss
        # scheduler
        if cfg["scheduler_version"]:
            scheduler.step(train_loss)
        # validation
        valid_loss = 0
        valid_dice = 0
        for id_ in valid_id:
            valid_dice_id, valid_loss_id = valid(model, dataloader[id_], criterion,
                                                 dice_ths=cfg["dice_ths"], device=cfg["device"], verbose=id_)
            valid_dice += valid_dice_id
            valid_loss += valid_loss_id
        valid_loss /= len(valid_id)
        valid_dice /= len(valid_id)
        monitor_epoch["valid_loss"] = valid_loss
        monitor_epoch["valid_dice"] = valid_dice

        print(f"Train-loss: {train_loss:.5} \t Valid-loss: {valid_loss:.5} \t Valid-dice: {valid_dice:.3}")
        print("-" * 70)

        # saving progress info
        monitor.append(monitor_epoch)
        dat.pkl_dump(pd.DataFrame(monitor), results_dir / "monitor.pkl")

        # saving weights - max DICE
        if dice_max is None:
            dice_max = valid_dice
        if valid_dice >= dice_max:
            dice_max = valid_dice
            torch.save(model.state_dict(), results_dir / "model_best_dice.pt")

        # saving weights - min LOSS
        if loss_min is None:
            loss_min = valid_loss
        # loss improvement
        if valid_loss <= loss_min:
            loss_min = valid_loss
            epochs_without_improvement = 0
            # save model
            torch.save(model.state_dict(), results_dir / "model_best_loss.pt")
            torch.save(optimizer.state_dict(), results_dir / "optimizer.pt")
        # -- no loss improvement
        else:
            epochs_without_improvement += 1

        # early stopping
        if epochs_without_improvement >= cfg["early_stopping"]:
            print("EARLY STOPPING!")
            break


if __name__ == "__main__":
    main(CONFIG)
