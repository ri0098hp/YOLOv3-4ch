import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets_multi import *
from utils.utils import *

save_folder = os.path.join("share", "test", "")


def test(
    cfg,
    data,
    weights=None,
    batch_size=16,
    imgsz=416,
    conf_thres=0.001,
    iou_thres=0.6,  # for nms
    save_json=False,
    single_cls=False,
    augment=False,
    model=None,
    dataloader=None,
    multi_label=True,
    plot=False,
    device="0",
):
    # Initialize/load model and set device
    if model is None:
        is_training = False

        # use these 3 lines for 5 layer-small
        # no_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch_utils.select_device(device, batch_size=batch_size)
        verbose = "test"

        # use this for 3 and 4 YOLO layer
        # device = torch_utils.select_device(opt.device, batch_size=batch_size)
        # verbose = opt.task == 'test'

        # Initialize model
        model = Darknet(cfg, imgsz)

        # Load weights
        attempt_download(weights)
        if weights.endswith(".pt"):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)["model"])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != "cpu" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    module_defs = parse_model_cfg(cfg)
    nchannels = module_defs[0]["channels"]  # parsing channel from cfg
    nc = 1 if single_cls else int(data["classes"])  # number of classes
    data_path = data["data_path"]
    rgb_folder = data["rgb_folder"]
    fir_folder = data["fir_folder"]
    labels_folder = data["labels_folder"]
    names = load_classes(data["names"])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(
            "test",
            data_path,
            rgb_folder,
            fir_folder,
            labels_folder,
            nchannels,
            imgsz,
            batch_size,
            rect=True,
            single_cls=opt.single_cls,
            pad=0.5,
        )
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    seen = 0
    model.eval()
    _ = model(torch.zeros((1, nchannels, imgsz, imgsz), device=device)) if device.type != "cpu" else None  # run once

    coco91class = coco80_to_coco91_class()
    s = ("%20s" + "%10s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP@0.5", "F1")
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        if batch_i != 50:  # manual
            continue
        imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split("_")[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append(
                        {
                            "image_id": image_id,
                            "category_id": coco91class[int(p[5])],
                            "bbox": [round(x, 3) for x in b],
                            "score": round(p[4], 5),
                        }
                    )

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            # if len(targets) > 20 and plot:
            if batch_i == 104:
                # ground truth
                f = save_folder + f"test_batch{batch_i}_gt.jpg"
                _ = plot_images(imgs, targets, paths=paths, names=names, fname=f)
                # predict
                f = save_folder + f"test_batch{batch_i}_pred.jpg"
                _ = plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plot, save_dir=save_folder[:-1], names=names)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    with open(save_folder + "test_results.txt", "w") as f:
        pf = ("%20s" + "%10s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP@0.5", "F1")
        print(pf, file=f)
        pf = ("%20s" + "%10.3g" * 6) % ("all", seen, nt.sum(), mp, mr, map, mf1)
        print(pf)
        print(pf, file=f)
        pf = ("%20s" + "%10.3g" * 2) % ("TP, FP", tp, fp)
        print(pf)
        print(pf, file=f)

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print("Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g" % t)

    # Save JSON
    if save_json and map and len(jdict):
        print("\nCOCO mAP with pycocotools...")
        imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
        with open("results.json", "w") as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob("../coco/annotations/instances_val*.json")[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes("results.json")  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except Exception:
            print(
                "WARNING: pycocotools must be installed with numpy==1.17 to run correctly. "
                "See https://github.com/cocodataset/cocoapi/issues/356"
            )

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # Clearing memory
    del model
    torch.cuda.empty_cache()
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument("--cfg", type=str, default="cfg/yolov3-spp-1cls-4channel.cfg", help="*.cfg path")
    parser.add_argument("--data", type=str, default="data/fujinolab-all.data", help="*.data path")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="weights path")
    parser.add_argument("--batch-size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IOU threshold for NMS")
    parser.add_argument("--save-json", action="store_true", help="save a cocoapi-compatible JSON results file")
    parser.add_argument("--task", default="test", help="'test', 'study', 'benchmark'")
    parser.add_argument("--device", default="0", help="device id (i.e. 0 or 0,1) or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ["coco.data", "coco2014.data", "coco2017.data"]])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)
    # Remove previous results
    for f in (
        glob.glob(f"{save_folder}test_batch*.jpg")
        + glob.glob(f"{save_folder}*_curve.svg")
        + glob.glob(f"{save_folder}*_log.txt")
    ):
        os.remove(f)
    os.makedirs(save_folder, exist_ok=True)
    # task = 'test', 'study', 'benchmark'
    if opt.task == "test":  # (default) test normally
        results, maps = test(
            opt.cfg,
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            plot=True,
            device=opt.device,
        )

    elif opt.task == "benchmark":  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt("benchmark.txt", y, fmt="%10.4g")  # y = np.loadtxt('study.txt')
