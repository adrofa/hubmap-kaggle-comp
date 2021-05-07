# Store images into zarr database.

from modules.data import HubmapDataset

from tqdm import tqdm
import zarr
import albumentations as A
import gc


CONFIG = {
    "version": "v1",

    "data_dir": r"data/",
    "out_dir": r"outputs/",

    "scale": 1,
    "chunk_size": 1024,
}


def main(cfg):
    dat = HubmapDataset(cfg["data_dir"], cfg["out_dir"])
    store = zarr.DirectoryStore(dat.path.out / "zarr" / cfg["version"] / f"db.zarr")
    database = zarr.group(store=store, overwrite=False)

    for id_, _ in tqdm(list(dat.get_inf("train").iterrows())):

        database.create_group(id_)
        img = dat.get_img(id_)
        msk = dat.get_msk(id_, "target")

        shape = dat.get_shape(id_)
        rescale = A.Resize(
            height=int(shape[0] / cfg["scale"]),
            width=int(shape[1] / cfg["scale"]),
            p=1.0
        )

        transformed = rescale(image=img, mask=msk)
        img, msk = transformed["image"], transformed["mask"]
        del transformed
        gc.collect()

        database[id_]["img"] = zarr.array(img, chunks=(cfg["chunk_size"], cfg["chunk_size"], 3))
        database[id_]["target"] = zarr.array(msk, chunks=(cfg["chunk_size"], cfg["chunk_size"]))
        del img, msk
        gc.collect()


if __name__ == "__main__":
    main(CONFIG)
