import albumentations as A


def get_augmentation(version):
    if version == "v1":
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v2":
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),
                A.OneOf([
                    A.RandomCrop(height=256, width=256, p=1),
                    A.RandomCrop(height=384, width=384, p=1),
                    A.RandomCrop(height=512, width=512, p=1),
                    A.RandomCrop(height=640, width=640, p=1),
                ], p=0.5),


                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v3":
        return {
            "train": A.Compose([
                A.RandomCrop(height=1024, width=1024, p=1),
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v4":
        # for 3x-scaled dataset | 256x256
        return {
            "train": A.Compose([
                A.RandomCrop(height=512, width=512, p=1),
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v5":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.RandomCrop(height=512, width=512, p=1),
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v6":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                     always_apply=False, p=0.25),
                A.RandomBrightness(limit=0.4, always_apply=False, p=0.25),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.25),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=50,
                               shadow_dimension=5, always_apply=False, p=0.25),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v7":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.RandomCrop(height=768, width=768, p=1),
                A.Resize(height=256, width=256, p=1.0),

                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                     always_apply=False, p=0.25),
                A.RandomBrightness(limit=0.4, always_apply=False, p=0.25),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.25),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=50,
                               shadow_dimension=5, always_apply=False, p=0.25),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)

            ]),
            "valid": A.Compose([
                A.Resize(height=256, width=256, p=1.0),
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v8":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.Rotate(p=0.5),
                A.RandomCrop(height=512, width=512, p=1),

                A.Flip(p=0.5),

                # color
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                     always_apply=False, p=0.25),
                A.RandomBrightness(limit=0.4, always_apply=False, p=0.25),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.25),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=50,
                               shadow_dimension=5, always_apply=False, p=0.25),

                # transform
                A.ElasticTransform(p=1, alpha=120, sigma=6, alpha_affine=0.25),
                A.GridDistortion(p=0.25),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.25),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v9":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                     always_apply=False, p=0.25),
                A.RandomBrightness(limit=0.4, always_apply=False, p=0.25),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.25),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=50,
                               shadow_dimension=5, always_apply=False, p=0.25),

                # transform
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=6, alpha_affine=0.25, p=0.25),
                    A.GridDistortion(p=0.25),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.25)
                ], p=0.75),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v10":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                # size
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=128, width=128, p=1.0),
                    A.CropNonEmptyMaskIfExists(height=256, width=256, p=1.0),
                    A.CropNonEmptyMaskIfExists(height=384, width=384, p=1.0)
                ], p=0.25),
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=4, p=1),

                # array shuffle
                A.OneOf([
                    A.MaskDropout(p=1),
                    A.RandomGridShuffle(p=1)
                ], p=0.5),

                # quality
                A.Downscale(scale_min=0.25, scale_max=0.75, interpolation=0, always_apply=False, p=0.5),
                A.GaussNoise(p=0.5),
                A.OneOf([
                    A.GlassBlur(p=1),
                    A.GaussianBlur(p=1),
                ], p=0.5),

                # colors
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                     always_apply=False, p=0.5),
                A.RandomBrightness(limit=0.4, always_apply=False, p=0.5),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=50,
                               shadow_dimension=5, always_apply=False, p=0.5),
                A.CLAHE(p=0.5),
                A.Equalize(p=0.5),
                A.ChannelShuffle(p=0.5),

                # transform
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=6, alpha_affine=0.25, p=1),
                    A.GridDistortion(p=1),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0.5),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v11":
        # custom normalization (see other/img_normalization.py)
        return {
            "train": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    elif version == "v12":
        # for 3x-scaled dataset | 512x512
        return {
            "train": A.Compose([
                A.Flip(p=0.5),
                A.Rotate(p=0.5),

                # size / quality
                A.OneOf([
                    A.CropNonEmptyMaskIfExists(height=128, width=128, p=1.0),
                    A.CropNonEmptyMaskIfExists(height=256, width=256, p=1.0),
                    A.CropNonEmptyMaskIfExists(height=384, width=384, p=1.0),
                    A.Downscale(scale_min=0.25, scale_max=0.75, interpolation=0, always_apply=False, p=1.0),
                ], p=0.25),
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=4, p=1),

                # array shuffle
                A.OneOf([
                    A.MaskDropout(p=1),
                    A.RandomGridShuffle(p=1)
                ], p=0.15),

                # noise
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GlassBlur(p=1),
                    A.GaussianBlur(p=1),
                ], p=0.15),

                # colors
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                     always_apply=False, p=0.15),
                A.RandomBrightness(limit=0.4, always_apply=False, p=0.15),
                A.RandomContrast(limit=0.2, always_apply=False, p=0.15),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=50,
                               shadow_dimension=5, always_apply=False, p=0.15),
                A.OneOf([
                    A.CLAHE(p=1),
                    A.Equalize(p=1),
                    A.ChannelShuffle(p=1),
                ], p=0.15),

                # transform
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=6, alpha_affine=0.25, p=1),
                    A.GridDistortion(p=1),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
                ], p=0.15),

                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
            "valid": A.Compose([
                A.Normalize(mean=(0.623, 0.520, 0.650), std=(0.278, 0.305, 0.274), max_pixel_value=255.0, p=1.0)
            ]),
        }

    else:
        raise Exception(f"Augmentation version '{version}' is UNKNOWN!")
