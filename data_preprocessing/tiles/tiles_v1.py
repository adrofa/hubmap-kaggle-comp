from modules.data import HubmapDataset

import pandas as pd
import zarr


CONFIG = {
    "version": "v1",

    "data_dir": r"data/",
    "out_dir": r"outputs/",
    "zarr_db_dir": r"outputs/zarr/v1/db.zarr",

    "valid_id": [
        "4ef6695ce",
        "0486052bb",
        "aaa6a05cc",
    ],

    "tile_size": 512,
    "overlap": 0.2,

    "train_valid_sample_size": 0.1,
    "seed": 0,
}


def main(cfg):
    dat = HubmapDataset(cfg["data_dir"], cfg["out_dir"])
    db = zarr.open_group(store=zarr.DirectoryStore(cfg["zarr_db_dir"]), mode="r")

    inf = dat.get_inf("train")
    train_id = [id_ for id_ in inf.index if id_ not in cfg["valid_id"]]
    valid_id = cfg["valid_id"]

    tile_dct = {}
    tile_size = (cfg["tile_size"], cfg["tile_size"])
    overlap = (cfg["overlap"], cfg["overlap"])

    # valid_dct
    tile_dct["valid_dct"] = {}
    for id_ in valid_id:
        df = pd.DataFrame()
        shape = db[id_]["target"].shape
        df["tiles"] = dat.gen_tiles(shape, tile_size, (0, 0), "shift_back")
        df["id"] = id_
        tile_dct["valid_dct"][id_] = df

    # train_df
    tile_dct["train_df"] = pd.DataFrame()
    for id_ in train_id:
        shape = db[id_]["target"].shape
        df = pd.DataFrame()
        df["tiles"] = dat.gen_tiles_v2(shape, tile_size, overlap)
        df["id"] = id_
        tile_dct["train_df"] = tile_dct["train_df"].append(df, ignore_index=True)

    # train_valid_df
    tile_dct["train_valid_df"] = tile_dct["train_df"].copy()
    tile_dct["train_valid_df"] = tile_dct["train_valid_df"].sample(
        frac=cfg["train_valid_sample_size"], random_state=cfg["seed"])

    # save results
    dat.pkl_dump(tile_dct, dat.path.out / "tiles" / cfg["version"] / "tile_dct.pkl")


if __name__ == "__main__":
    main(CONFIG)
