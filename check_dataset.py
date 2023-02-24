import glob
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pprint

import yaml

from utils.datasets import create_dataloader
from utils.general import check_dataset
from utils.torch_utils import torch_distributed_zero_first

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
os.chdir(ROOT)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))


def main():
    data_dict = None
    pprint(glob.glob("data/*.yaml"))
    data_name = input("name of data: ")
    if os.path.exists(f"data/hyps/{data_name}.yaml"):
        with open(f"data/hyps/{data_name}.yaml", errors="ignore") as f:
            hyp = yaml.safe_load(f)
    else:
        with open("data/hyps/default.yaml", errors="ignore") as f:
            hyp = yaml.safe_load(f)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(f"data/{data_name}.yaml")  # check if None
    data_path = data_dict["data_path"]
    rgb_dir, fir_dir = data_dict["rgb_folder"], data_dict["fir_folder"]
    labels_dir = data_dict["labels_folder"]
    nch = data_dict["ch"]
    with open("data/custom.yaml", "w") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

    if "pos_imgs_train" in hyp.keys():
        print("pos_imgs_train:", hyp["pos_imgs_train"])
    else:
        print("pos_imgs_train:", "all")
    if "neg_ratio_train" in hyp.keys():
        print("neg_ratio_train:", hyp["neg_ratio_train"])
    else:
        print("neg_ratio_train:", "all")

    while True:
        try:
            train_loader, dataset = create_dataloader(
                "train",
                data_path,
                rgb_dir,
                fir_dir,
                labels_dir,
                nch,
                640,
                1,
                32,
                single_cls=True,
                hyp=hyp,
                augment=True,
                cache=False,
                rect=False,
                rank=-1,
                workers=1,
                image_weights=False,
                quad=False,
                prefix="train: ",
                shuffle=True,
            )
        except PermissionError:
            pass

        with open(f"data/hyps/{data_name}.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)

        # input number
        print("0: 決定, -1: 全データ, 自然数: ラベル有画像数")
        p = int(input("pos_imgs_train: "))
        if p == 0:
            break
        n = float(input("neg_ratio_train: "))

        # apply to dict
        if p != -1:
            hyp["pos_imgs_train"] = p
        elif "pos_imgs_train" in hyp.keys():
            del hyp["pos_imgs_train"]
        if n != -1:
            hyp["neg_ratio_train"] = n
        elif "neg_ratio_train" in hyp.keys():
            del hyp["neg_ratio_train"]

    if "pos_imgs_val" in hyp.keys():
        print("pos_imgs_val:", hyp["pos_imgs_val"])
    else:
        print("pos_imgs_val:", "all")
    if "neg_ratio_val" in hyp.keys():
        print("neg_ratio_val:", hyp["neg_ratio_val"])
    else:
        print("neg_ratio_val:", "all")

    while True:
        try:
            train_loader, dataset = create_dataloader(
                "val",
                data_path,
                rgb_dir,
                fir_dir,
                labels_dir,
                nch,
                640,
                1,
                32,
                single_cls=True,
                hyp=hyp,
                augment=True,
                cache=False,
                rect=False,
                rank=-1,
                workers=1,
                image_weights=False,
                quad=False,
                prefix="val: ",
                shuffle=True,
            )
        except PermissionError:
            pass

        with open(f"data/hyps/{data_name}.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)

        print("0: 決定, -1: 全データ, 0.1~1.0: ラベル無しデータの割合")
        p = int(input("pos_imgs_val: "))
        if p == 0:
            break
        n = float(input("neg_ratio_val: "))
        # apply to dict
        if p != -1:
            hyp["pos_imgs_val"] = p
        elif "pos_imgs_val" in hyp.keys():
            del hyp["pos_imgs_val"]
        if n != -1:
            hyp["neg_ratio_val"] = n
        elif "neg_ratio_val" in hyp.keys():
            del hyp["neg_ratio_val"]


if __name__ == "__main__":
    mp.freeze_support()
    main()
