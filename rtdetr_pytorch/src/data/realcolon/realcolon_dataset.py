import random

import torch
from src.core import register
from src.data.coco.coco_dataset import CocoDetection

__all__ = ["RealColonDataset"]


@register
class RealColonDataset(torch.utils.data.Dataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        remap_mscoco_category=False,
        neg_ratio=0.0,
    ):
        self.ds = CocoDetection(
            img_folder, ann_file, transforms, return_masks, remap_mscoco_category
        )
        self.neg_ratio = neg_ratio

        self.indexes = None

        self._build_indexes()

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        return self.ds[index]

    def resample_negatives(self):
        self._build_indexes()

    def _build_indexes(self):
        self.indexes = []  # idx -> sample id

        posidx = []
        negidx = []

        for i in range(len(self.ds)):
            anns = get_target_ann(self.ds, get_id(self.ds, i))

            if len(anns) > 0:
                posidx.append(i)
            else:
                negidx.append(i)

        self.indexes.extend(posidx)

        n_neg = int(len(negidx) * self.neg_ratio)
        n_neg = min(n_neg, len(negidx))

        negidx = random.sample(negidx, n_neg)

        self.indexes.extend(negidx)


def get_id(ds: CocoDetection, idx: int):
    return ds.ids[idx]


def get_target_ann(ds: CocoDetection, sample_id: int):
    return ds._load_target(sample_id)


def get_image_ann(ds: CocoDetection, sample_id: int):
    return ds.coco.loadImgs(sample_id)[0]


def get_image(ds: CocoDetection, sample_id: int):
    return ds._load_image(sample_id)


if __name__ == "__main__":
    ds = RealColonDataset(
        "dataset/real_colon_dataset_coco_fmt_3subsets_poslesion-1_negratio1/train_images",
        "dataset/real_colon_dataset_coco_fmt_3subsets_poslesion-1_negratio1/train_ann.json",
        transforms=None,
        return_masks=False,
        remap_mscoco_category=False,
        neg_ratio=0.1,
    )

    print(len(ds))

    ds.resample_negatives()

    print(len(ds))
