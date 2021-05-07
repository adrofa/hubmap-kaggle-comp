import pandas as pd
import numpy as np

import os
from pathlib import Path
import itertools

import json
import tifffile
import pickle

import mahotas


class HubmapPath:
    """Paths to HuBMAP dataset and outputs directory.

    Args:
        data_dir: directory with HuBMAP files.
        out_dir: directory to store outputs.
    """
    def __init__(self, data_dir, out_dir):
        self.data = Path(data_dir)
        self.info = self.data / "HuBMAP-20-dataset_information.csv"
        self.annotation = self.data / "train.csv"
        self.submission = self.data / "sample_submission.csv"
        self.out = Path(out_dir)
        os.makedirs(self.out, exist_ok=True)


class HubmapDataset:

    def __init__(self, data_dir, out_dir):
        self.path = HubmapPath(data_dir, out_dir)
        self.inf = self.init_inf()
        self.anatomy_types = self.init_anatomy_types()

    def init_inf(self):
        """Open information csv-file, and add 'split' (train/test) column."""
        df = pd.read_csv(self.path.info)

        # train/test split
        train_files = [f + ".tiff" for f in pd.read_csv(self.path.annotation)["id"].to_list()]
        df["split"] = df["image_file"].apply(lambda x: "train" if x in train_files else "test").values

        # id as index
        df["id"] = df["image_file"].apply(lambda x: x.split(".")[0])
        df.set_index("id", drop=True, inplace=True)

        return df

    def init_anatomy_types(self):
        """Get unique anatomical structures' names from anatomical_structures_segmention_files"""
        anatomy_dct = self.get_anatomy_dct()
        anatomy_types = [plg["properties"]["classification"]["name"]
                         for id_ in anatomy_dct
                         for plg in anatomy_dct[id_]]
        anatomy_types = list(set(anatomy_types))
        anatomy_types.sort()
        return anatomy_types

    def get_anatomy_dct(self):
        """Collect data_preprocessing of anatomical_structures_segmention_files into dict (key = id)."""
        anatomy_jsn_dct = {}
        for id_, row in self.inf.iterrows():
            jsn = self.jsn_load(self.path.data / row["split"] / row["anatomical_structures_segmention_file"])
            anatomy_jsn_dct[id_] = jsn
        return anatomy_jsn_dct

    def get_inf(self, type_):
        """Returns pandas.DataFrame with train/test images or info for the provided image_id."""
        if type_ in ["train", "test"]:
            return self.inf[self.inf["split"] == type_].copy()
        elif type_ in self.inf.index.to_list():
            return self.inf.loc[type_]
        else:
            raise Exception(f"'{type}' for inf is UNKNOWN!")

    def get_img(self, id_):
        """Load image for the provided image_id."""
        inf = self.get_inf(id_)
        return self.tiff2numpy(self.path.data / inf["split"] / inf["image_file"])

    @staticmethod
    def tiff2numpy(tiff_file):
        """Load tiff file as a numpy array."""
        img = tifffile.imread(tiff_file).squeeze()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return img

    def get_msk(self, id_, type_):
        """ Load mask for the provided image id. Available masks:
        target, target_plg and anatomy_types.
        """
        if type_ == "target":
            ann = pd.read_csv(self.path.annotation).set_index("id")["encoding"]
            if id_ not in ann:
                raise Exception(f"No annotation for id {id_}!")
            return self.rle2msk(ann[id_], self.get_shape(id_))

        elif type_ in ["target_plg"] + self.anatomy_types:
            shape = self.get_shape(id_)
            coordinates = self.get_coordinates(id_, type_)
            if len(coordinates) == 0:
                return None
            else:
                return self.crd2msk(shape, coordinates)

        else:
            raise Exception(f"'{type_}' for msk is UNKNOWN!")

    def get_coordinates(self, id_, type_):
        """Get polygons coordinates for the provided id_ and mask type."""
        inf = self.get_inf(id_)
        if type_ == "target_plg":
            jsn_file = self.path.data / inf["split"] / inf["glomerulus_segmentation_file"]
            if not os.path.isfile(jsn_file):
                raise Exception(f"No annotation for id {id_}!")
            jsn = self.jsn_load(jsn_file)
            coordinates = [np.array(elem["geometry"]["coordinates"]).squeeze() for elem in jsn]

        elif type_ in self.anatomy_types:
            jsn = self.jsn_load(self.path.data / inf["split"] / inf["anatomical_structures_segmention_file"])
            coordinates = []
            for plg in jsn:
                if plg["properties"]["classification"]["name"] == type_:
                    candidates = plg["geometry"]["coordinates"]
                    if len(candidates) == 1:
                        candidates = [candidates]
                    for cand in candidates:
                        coordinates.append(np.array(cand).squeeze())

        else:
            raise Exception(f"'{type_}' for coordinates in UNKNOWN!")

        return coordinates

    @staticmethod
    def crd2msk(shape, coordinates):
        """Transform polygons into binary numpy array."""
        canvas = np.zeros(shape, dtype='uint8').T
        for elem in coordinates:
            mahotas.polygon.fill_polygon(elem.astype(int), canvas)
        return canvas.T

    def get_shape(self, id_):
        """Get image shape. Returns tuple of length 2 (height x width)."""
        inf = self.get_inf(id_)
        return inf["height_pixels"], inf["width_pixels"]

    @staticmethod
    def rle2msk(rle, shape):
        """ Modified function from: https://www.kaggle.com/matjes/hubmap-efficient-sampling-deepflash2-train
        Replaced positions in width and height => shape=(height, width)
        """
        s = rle.split()
        starts, lengths = [
            np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
        ]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype='uint8')
        for lo, hi in zip(starts, ends):
            img[lo: hi] = 1
        img = img.reshape((shape[1], shape[0])).T
        return img

    @staticmethod
    def msk2rle(msk):
        """ from: https://www.kaggle.com/matjes/hubmap-efficient-sampling-deepflash2-sub
        (original name: rle_encode_less_memory)
        """
        # the image should be transposed
        pixels = msk.T.flatten()

        # This simplified method requires first and last pixel to be zero
        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] -= runs[::2]

        return ' '.join(str(x) for x in runs)

    @staticmethod
    def pkl_dump(obj, file):
        """Dump object with pickle."""
        if os.path.dirname(file) != '':
            os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def pkl_load(file):
        """Load object from pickle."""
        with open(file, "rb") as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def jsn_dump(dct, file):
        """Dump dict as json."""
        with open(file, "w") as f:
            json.dump(dct, f, sort_keys=True, indent=4)

    @staticmethod
    def jsn_load(file):
        """Load json file."""
        with open(file, "r") as f:
            jsn = json.load(f)
        return jsn

    @staticmethod
    def gen_tiles(shape, tile_size=(1024, 1024), shift=(0, 0), last="shift_back"):
        """Generate list of tiles. Tiles are slices of the original image of the provided shape.

        Args:
            shape (tuple): height x width of the image in pixelx.
            tile_size (tuple): tuple(height, width) of each tile in pixels.
            shift (tuple): tuple(height, width) number of pixels to shift tiles from the beginning.
            last (str): processing of the last tile
                 "shift_back" - last tile ends at the border of the image (shape);
                 "do_nothing" - last tile may overlap image borders;
                 "drop" - remove tiles, which overlap borders.

        Returns:
            coords (list): list of tuples (x1, x2, y1, y2).
        """

        for dim in [0, 1]:
            assert shift[dim] < tile_size[dim], f"Dim-{dim}: shift >= tile_size"

        def split_dim(shape_, tile_size_, shift_, last_):
            dim_ = [(i, i + tile_size_)
                    for i in range(shift_, shape_, tile_size_)]
            if last_ == "do_nothing":
                pass
            elif last_ == "shift_back":
                dim_[-1] = (shape_ - tile_size_, shape_)
            elif last_ == "drop":
                dim_ = dim_[:-1]
            else:
                raise Exception(f"Last '{last}' is UNKNOWN!")

            return dim_

        rows = split_dim(shape[0], tile_size[0], shift[0], last)
        cols = split_dim(shape[1], tile_size[1], shift[1], last)

        coords = [tuple([elem for tpl in coords_raw for elem in tpl])
                  for coords_raw in itertools.product(rows, cols)]

        return coords

    @staticmethod
    def gen_tiles_v2(shape, tile_size, overlap):
        """Generate list of tiles. Tiles are slices of the original image of the provided shape.
        The last tiles will be shifted back to not overlap the image borders.

        Args:
            shape (tuple): tuple(height, width) of the image in pixels.
            tile_size (tuple): tuple(height, width) of each tile in pixels.
            overlap (tuple): tuple(vertical, horizontal) percentage of tiles overlapping;
                0 <= vertical or horizontal <1

        Returns:
            coords (list): list of tuples (x1, x2, y1, y2).
        """

        def split_dim(shape_, tile_size_, overlap_):
            overlap_px = int(round(tile_size_ * overlap_))

            tiles = []
            start = 0
            while True:
                end = start + tile_size_

                if end > shape_:
                    end = shape_
                    start = end - tile_size_

                tiles.append((start, end))

                if end == shape_:
                    break
                else:
                    start = end - overlap_px

            return tiles

        rows = split_dim(shape[0], tile_size[0], overlap[0])
        cols = split_dim(shape[1], tile_size[1], overlap[1])

        coords = [tuple([elem for tpl in coords_raw for elem in tpl])
                  for coords_raw in itertools.product(rows, cols)]

        return coords
