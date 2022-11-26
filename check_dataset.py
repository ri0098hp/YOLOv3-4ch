from utils.datasets import create_dataloader
import yaml
import multiprocessing as mp
import os
from utils.torch_utils import torch_distributed_zero_first
from utils.general import check_dataset
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
os.chdir(ROOT)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))


def main():
    data_dict = None
    data_name = input("name of data: ")
    if os.path.exists(f"data/hyps/custom-{data_name}.yaml"):
        with open(f"data/hyps/custom-{data_name}.yaml", errors="ignore") as f:
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
        p = int(input("pos_imgs_train: "))
        with open(f"data/hyps/custom-{data_name}.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        if p == 0:
            break
        hyp["pos_imgs_train"] = p

    if "pos_imgs_val" in hyp.keys():
        print("pos_imgs_val:", hyp["pos_imgs_val"])
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
        p = int(input("pos_imgs_val: "))
        with open(f"data/hyps/custom-{data_name}.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        if p == 0:
            break
        hyp["pos_imgs_val"] = p


if __name__ == "__main__":
    mp.freeze_support()
    main()
