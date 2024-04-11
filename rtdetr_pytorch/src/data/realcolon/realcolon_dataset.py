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
        self.ids = {}
        self._build_ids()

    def _build_ids(self):
        self.ids = {}  # idx -> sample id

        idx2posid = {}  # idx -> positive sample id
        idx2negid = {}  # idx -> negative sample id

        for i in range(len(self.ds)):
            sample_id = get_id(self.ds, i)
            anns = get_target_ann(self.ds, sample_id)

            if len(anns) > 0:
                idx2posid[i] = sample_id
            else:
                idx2negid[i] = sample_id

        self.ids.update(idx2posid)

        # add to ids neg_ratio % of negative, randomly
        n_neg = int(len(idx2negid) * self.neg_ratio)
        neg_idx = list(idx2negid.keys())
        neg_idx = random.sample(neg_idx, len(neg_idx))
        neg_idx = neg_idx[:n_neg]

        idx2negid = {i: idx2negid[i] for i in neg_idx}

        self.ids.update(idx2negid)


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
