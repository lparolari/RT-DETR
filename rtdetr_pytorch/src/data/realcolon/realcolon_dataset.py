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

        self.ids = None

        self._build_ids()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        anns = get_target_ann(self.ds, sample_id)
        img = get_image(self.ds, sample_id)

        return img, anns

    def resample_negatives(self):
        self._build_ids()

    def _build_ids(self):
        self.ids = []  # idx -> sample id

        posid = []
        negid = []

        for i in range(len(self.ds)):
            sample_id = get_id(self.ds, i)
            anns = get_target_ann(self.ds, sample_id)

            if len(anns) > 0:
                posid.append(sample_id)
            else:
                negid.append(sample_id)

        self.ids.extend(posid)

        n_neg = int(len(negid) * self.neg_ratio)
        n_neg = min(n_neg, len(negid))

        negid = random.sample(negid, n_neg)

        self.ids.extend(negid)


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
