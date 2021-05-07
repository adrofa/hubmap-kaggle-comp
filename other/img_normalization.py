# Estimate mean and std of pixels values per channel
# for image normalization.

from modules.data import HubmapDataset

import numpy as np
import os
import zarr
from tqdm import tqdm


CONFIG = {
    "version": "v1",

    "data_dir": r"data/",
    "out_dir": r"outputs/",
    "zarr_db_dir": r"outputs/zarr/v1/db.zarr",

    "tiles_version": "v1",
}


def main(cfg):
    # data_preprocessing
    data_dir = cfg["data_dir"]
    out_dir = cfg["out_dir"]
    dat = HubmapDataset(data_dir, out_dir)
    # db_zarr
    db = zarr.open_group(store=zarr.DirectoryStore(cfg["zarr_db_dir"]), mode="r")

    # results dir
    results_dir = dat.path.out / "normalization" / cfg["version"]
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
    tiles = tile_dct["train_df"]

    # mean
    sum = np.zeros(3)
    N = (tiles["tile"].apply(lambda x: x[1] - x[0]) *
         tiles["tile"].apply(lambda x: x[3] - x[2])).sum()

    for _, row in tqdm(tiles.iterrows(), total=len(tiles)):
        id_ = row["id"]
        c = row["tile"]

        slc = np.s_[c[0]: c[1], c[2]: c[3]]
        img = db[id_]["img"][slc] / 255

        sum += img.sum(axis=(0, 1))

    mean = sum / N
    dat.pkl_dump(mean, results_dir / "mean.pkl")

    # std
    diff_squared = np.zeros(3)

    for _, row in tqdm(tiles.iterrows(), total=len(tiles)):
        id_ = row["id"]
        c = row["tile"]

        slc = np.s_[c[0]: c[1], c[2]: c[3]]
        img = db[id_]["img"][slc] / 255

        diff_squared += ((img - mean) ** 2).sum(axis=(0, 1))

    std = np.sqrt(diff_squared / N)
    dat.pkl_dump(std, results_dir / "std.pkl")

    print(f"MEAN: {mean}")
    print(f"STD: {std}")


if __name__ == "__main__":
    main(CONFIG)
