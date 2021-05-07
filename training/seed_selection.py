# Looking for good model's weights initialization.

from modules.data import HubmapDataset
from modules.training import seed_everything, TrainDataset, train, ValidDataset, valid
from training_params.model import get_model
from training_params.optimizer import get_optimizer
from training_params.criterion import get_criterion
from training_params.augmentation import get_augmentation

import random
import os
import zarr
import torch
from torch.utils.data import DataLoader
import time


CONFIG = {
    "version": "debug",
    "debug": False,
    "device": "cuda",

    "max_time": 15,  # minutes
    "epoch_num": 1,

    "data_dir": r"data/",
    "out_dir": r"outputs/",
    "zarr_db_dir": r"outputs/zarr/v1/db.zarr",

    "model_version": "unet_v2",
    "optimizer_version": "adam_v1",

    "criterion_version": "dice_v1",
    "dice_ths": 0.5,

    "tiles_version": "v1",
    "augmentation_version": "v12",

    "batch_size": 4,
    "fold": 0,
}


def main(cfg):
    dat = HubmapDataset(cfg["data_dir"], cfg["out_dir"])
    db = zarr.open_group(store=zarr.DirectoryStore(cfg["zarr_db_dir"]), mode="r")

    # results dir
    results_dir = dat.path.out / "seed_selection" / cfg["version"]
    if cfg["version"] == "debug":
        os.makedirs(results_dir, exist_ok=True)
    else:
        try:
            os.makedirs(results_dir, exist_ok=False)
        except:
            raise Exception(f"Version {cfg['version']} exists!")
    # save config
    dat.jsn_dump(cfg, results_dir / "config.json")

    # tiles
    tile_dct = dat.pkl_load(dat.path.out / "tiles" / cfg['tiles_version'] / "tile_dct.pkl")

    # train/valid split
    valid_dct = tile_dct["valid_dct"]
    valid_id = list(valid_dct.keys())
    train_df = tile_dct["train_df"]
    # -- debug (cutting data)
    if cfg["debug"]:
        valid_dct = {id_: valid_dct[id_].head(cfg["debug"]) for id_ in valid_id}
        train_df = train_df.head(cfg["debug"])

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = {id_: ValidDataset(db, valid_dct[id_], augmentation["valid"]) for id_ in valid_id}
    dataset["train"] = TrainDataset(db, train_df, augmentation["valid"])

    # PyTorch DataLoaders initialization
    dataloader = {
        id_: DataLoader(dataset[id_], batch_size=cfg["batch_size"], shuffle=True, num_workers=1)
        for id_ in valid_id
    }
    dataloader["train"] = DataLoader(dataset["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=1)

    # seed search
    time_start = time.time()
    time_stop = False
    seeds = []
    valid_loss_min = None
    while time_stop is not True:

        # seed value
        while True:
            seed = random.randint(0, 100_000)
            if seed not in seeds:
                seeds.append(seed)
                break
        seed_everything(seed)

        # model initialization
        model = get_model(cfg["model_version"])

        # loss-function
        criterion = get_criterion(cfg["criterion_version"])

        # training
        optimizer = get_optimizer(cfg["optimizer_version"], model.parameters())
        for epoch in range(cfg["epoch_num"]):
            train(model, dataloader["train"], criterion, optimizer)

        # validation
        valid_loss = 0
        for id_ in valid_id:
            _, valid_loss_id = valid(model,
                                     dataloader[id_],
                                     criterion,
                                     dice_ths=cfg["dice_ths"],
                                     device=cfg["device"],
                                     verbose=id_)
            valid_loss += valid_loss_id
        valid_loss /= len(valid_id)
        print(f"Seed-{seed} | Valid-loss: {valid_loss:.5}")

        # saving progress
        if valid_loss_min is None:
            valid_loss_min = valid_loss

        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), results_dir / "model.pt")
            with open(results_dir / "best_seed.txt", "w") as f:
                f.write(str(seed))

        # stop by time
        if (time.time() - time_start) / 60 >= cfg["max_time"]:
            time_stop = True


if __name__ == "__main__":
    main(CONFIG)
