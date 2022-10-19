# loading grayscale images modified from
# https://github.com/pieterbl86/yolov3/commit/18789e8e1d2c24e0cfb4b2f638fc3e66048d954e

import glob
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xywh2xyxy, xyxy2xywh

help_url = "https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data"
img_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".dng"]
vid_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


class LoadImages:  # for inference
    def __init__(self, path, nchannel, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        RGB_path = path + "/RGB/"
        IR_path = path + "/FIR/"
        print(path)
        if os.path.isdir(path):
            if nchannel == 4:
                RGB_file = sorted(glob.glob(os.path.join(RGB_path, "*.*")))
                IR_file = sorted(glob.glob(os.path.join(IR_path, "*.*")))
            else:
                files = sorted(glob.glob(os.path.join(path, "*.*")))
        elif os.path.isfile(path):
            files = [path]
        else:
            raise ValueError(f"{path} is not existing")

        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nV = len(videos)
        if nchannel == 4:
            images_RGB = [x for x in RGB_file if os.path.splitext(x)[-1].lower() in img_formats]
            images_IR = [x for x in IR_file if os.path.splitext(x)[-1].lower() in img_formats]
            # a pair, so not counting 2 different images, assuming no videos
            nI = len(images_RGB)
            self.img_RGB = images_RGB
            self.img_IR = images_IR
        else:
            images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
            nI = len(images)
            self.files = images + videos

        self.nchannel = nchannel
        self.img_size = img_size
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = "images"
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, "No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s" % (
            path,
            img_formats,
            vid_formats,
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration

        if self.nchannel == 4:
            path_RGB = self.img_RGB[self.count]
            path_IR = self.img_IR[self.count]
            # print("path inside", path_RGB)
            # print("path inside", path_IR)
        else:
            path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print("video %g/%g (%g/%g) %s: " % (self.count + 1, self.nF, self.frame, self.nframes, path), end="")

        else:
            # Read image
            self.count += 1
            if self.nchannel == 1:
                img0 = cv2.imread(path, 0)  # grayscale
            elif self.nchannel == 4:
                img_rgb = cv2.imread(path_RGB)
                img_ir = cv2.imread(path_IR, 0)

                # merge the file for 4 channel
                img0 = cv2.merge((img_rgb, img_ir))
            else:
                img0 = cv2.imread(path)  # BGR / 3 channel

            if self.nchannel == 4:
                assert img0 is not None, "Image Not Found " + path_RGB
                print("image %g/%g %s: " % (self.count, self.nF, path_RGB), end="")

                assert img0 is not None, "Image Not Found " + path_IR
                print("image %g/%g %s: " % (self.count, self.nF, path_IR), end="")
            else:
                assert img0 is not None, "Image Not Found " + path
                print("image %g/%g %s: " % (self.count, self.nF, path), end="")

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path_RGB, path_IR, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources="streams.txt", img_size=416):
        self.mode = "images"
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print("%g/%g: %s... " % (i + 1, n, s), end="")
            cap = cv2.VideoCapture(0 if s == "0" else s)
            assert cap.isOpened(), "Failed to open %s" % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(" success (%gx%g at %.2f FPS)." % (w, h, fps))
            thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        # rect inference if all shapes equal
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        # BGR to RGB, to bsx3x416x416
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        is_train,
        data_path,
        rgb_folder,
        fir_folder,
        labels_folder,
        nchannel=3,
        img_size=416,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        pad=0.0,
    ):
        print()
        # Define image files
        path = str(Path(data_path))
        # print(f"target: {path}")
        os.makedirs(path + os.sep + "cache", exist_ok=True)
        try:
            # loading RGB images
            f = sorted(glob.iglob(os.path.join(path, "**", rgb_folder, "*.*"), recursive=True))
            f = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
            f.sort(key=lambda s: int(re.search(r"(\d+)\.", s).groups()[0]))
            sep = f[::5]  # dividing train:val = 4:1
            if is_train == "train":
                for target in sep:
                    f.remove(target)
                self.img_files = f
            else:
                self.img_files = sep
        except Exception:
            raise Exception(f"Error loading data from {path}. See {help_url}")

        try:
            # loading FIR images
            # f = sorted(glob.iglob(os.path.join(path, "**", fir_folder, "*.*"), recursive=True))
            # f = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
            # f.sort(key=lambda s: int(re.search(r"(\d+)\.", s).groups()[0]))
            # sep = f[::7]
            # if is_train == "train":
            #     for target in sep:  # dividing train:val = 6:1
            #         f.remove(target)
            #     self.img_files_ir = f
            # else:
            #     self.img_files_ir = sep
            self.img_files_ir = [
                x.replace(os.sep + rgb_folder + os.sep, os.sep + fir_folder + os.sep) for x in self.img_files
            ]
        except Exception:
            raise Exception(f"Error loading data from {path}. See {help_url}")

        # show miss match files between rgb and fir
        miss_match = [
            os.path.basename(rgb)
            for rgb, fir in zip(self.img_files, self.img_files_ir)
            if os.path.basename(rgb) != os.path.basename(fir)
        ]
        if miss_match != []:
            print(f"miss matched imgaes :{len(miss_match)}")

        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.nchannel = nchannel
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        print(f"rect: {self.rect}")
        # load 4 images at a time into a mosaic (only during training)
        self.mosaic = self.augment and not self.rect
        print(f"mosaic: {self.mosaic}")

        # Define labels
        self.label_files = []
        for x in self.img_files_ir:
            x = x.replace(os.sep + "FIR" + os.sep, os.sep + labels_folder + os.sep)
            label_fp = x.replace(os.path.splitext(x)[-1], ".txt")
            if os.path.exists(label_fp):  # labels in FIR_labels
                self.label_files.append(label_fp)
            else:  # labels in same folder
                self.label_files.append(x.replace(os.path.splitext(x)[-1], ".txt"))

        # Read image shapes (wh)
        sp = os.path.join(path.replace(".txt", ""), "cache", f"{is_train}.shapes")  # shapefile path
        try:
            with open(sp, "r") as f:  # read existing shapefile
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == n, "Shapefile out of sync"
        except Exception:
            s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc="Reading image shapes")]
            np.savetxt(sp, s, fmt="%g")  # overwrites existing (if any)

        self.shapes = np.array(s, dtype=np.float64)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.img_files_ir = [self.img_files_ir[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.0 + pad).astype(np.int) * 32

        # Cache labels
        self.imgs = [None] * n
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        # number missing, found, empty, datasubset, duplicate
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0
        # saved labels in *.npy file
        np_labels_path = os.path.join(str(Path(self.label_files[0]).parents[2]), "cache", f"{is_train}_labels.npy")
        if os.path.isfile(np_labels_path):
            s = np_labels_path  # print string
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                print("here")
                self.labels = x
                labels_loaded = True
                print(f"npy at {s} was loaded")
        else:
            # save npy folder
            s = path.replace("images", "labels")
            print("there is no labels cache")

        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            if labels_loaded:
                l = self.labels[i]
            else:
                try:
                    with open(file, "r") as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception:
                    nm += 1
                    continue

            if l.shape[0]:
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).all(), "negative labels: %s" % file
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                    nd += 1
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1e4:
                    if ns == 0:
                        create_folder(path="./datasubset")
                        os.makedirs("./datasubset/images")
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open("./datasubset/images.txt", "a") as f:
                            f.write(self.img_files[i] + "\n")

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            # make new output folder
                            os.makedirs(Path(f).parent)

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        # clip boxes outside of image
                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1] : b[3], b[0] : b[2]]), "Failure extracting classifier boxes"
            else:
                # print('empty labels for image %s' % self.img_files[i])  # file empty
                ne += 1
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = "Caching labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                s,
                nf,
                nm,
                ne,
                nd,
                n,
            )

        # show avaival data sizes
        print("##########################")
        print(f"{is_train} data has ...")
        print(f"RGB: {len(self.img_files)} files")
        print(f"FIR: {len(self.img_files_ir)} files")
        print(f"labes: {nf} files")
        print("##########################")

        assert nf > 0 or n == 20288, "No labels found in %s. See %s" % (os.path.dirname(file) + os.sep, help_url)
        if not labels_loaded and n > 1000:
            # print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = "Caching images (%.1fGB)" % (gb / 1e9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image

            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except Exception:
                    print("Corrupted image detected: %s" % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index, self.nchannel)
            shapes = None

        else:
            # Load image
            if self.nchannel == 4:
                img, (h0, w0), (h, w) = load_image_multi(self, index)
            else:
                img, (h0, w0), (h, w) = load_image(self, index, self.nchannel)  # RGB or IR

            # Letterbox
            # final letterboxed shape
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            # print("Image shape after letterbox:", img.shape)
            # for COCO mAP rescaling
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                )

            # Augment colorspace
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        if img.ndim == 2:
            # grayscale add another dimension for channel
            img = np.expand_dims(img, axis=2)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img.copy()), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_image_multi(self, index):
    img = self.imgs[index]
    if img is None:
        path_rgb = self.img_files[index]
        path_ir = self.img_files_ir[index]

        img_rgb = cv2.imread(path_rgb)  # reading rgb
        img_ir = cv2.imread(path_ir, 0)  # reading grayscale
        img = cv2.merge((img_rgb, img_ir))  # combine rgb + ir

        assert img is not None, "Image Not Found " + path_rgb
        h0, w0 = img.shape[:2]  # orig hw #only 2 values, since grayscale
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            # print(f"Image after resize {img.shape}")
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        # img, hw_original, hw_resized
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def load_image(self, index, nchannel=3):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # 3 channels or 1 channel
    img = self.imgs[index]
    if img is None:  # not cached
        # checking channel
        if nchannel == 1:
            path = self.img_files_ir[index]
            # 1 channel (grayscale), expands the dimension as 1 channel doesn't have detail in img.shape cv2
            img = np.expand_dims(cv2.imread(path, 0), axis=2)
        else:
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR or 3 channel
        assert img is not None, "Image Not Found " + path
        # print(f"img.shape[:2] {img.shape[:2]}")
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        # img, hw_original, hw_resized
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    # enabled: okuda
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index, nchannel):
    # enabled : okuda
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 3 additional image indices
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    for i, index in enumerate(indices):
        # Load image
        # img, _, (h, w) = load_image(self, index)
        if nchannel == 4:
            img, _, (h, w) = load_image_multi(self, index)  # for 4 channels
        else:
            img, _, (h, w) = load_image(self, index, nchannel)  # for 3 or 1 channel
        # place img in img4
        if i == 0:  # top left
            # base image with 4 tiles
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            # xmin, ymin, xmax, ymax (large image)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            # xmin, ymin, xmax, ymax (small image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # img4[ymin:ymax, xmin:xmax]
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        # use with random_affine
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_affine(
        img4,
        labels4,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        border=-s // 2,
    )  # border to remove

    return img4, labels4


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_affine(img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, border=0):
    # enabled: okuda
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
            np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
        ).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


# from utils.datasets import *; reduce_img_size()
def reduce_img_size(path="../data/sm4/images", img_size=1024):
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + "_reduced"  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob("%s/*.*" % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            # .replace(Path(f).suffix, '.jpg')
            fnew = f.replace(path, path_new)
            cv2.imwrite(fnew, img)
        except Exception:
            print("WARNING: image failure %s" % f)


def convert_images2bmp():  # from utils.datasets import *; convert_images2bmp()
    # Save images
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    # for path in ['../coco/images/val2014', '../coco/images/train2014']:
    for path in ["../data/sm4/images", "../data/sm4/background"]:
        create_folder(path + "bmp")
        for ext in formats:  # ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            for f in tqdm(glob.glob("%s/*%s" % (path, ext)), desc="Converting %s" % ext):
                cv2.imwrite(f.replace(ext.lower(), ".bmp").replace(path, path + "bmp"), cv2.imread(f))

    # Save labels
    # for path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
    for file in ["../data/sm4/out_train.txt", "../data/sm4/out_test.txt"]:
        with open(file, "r") as f:
            lines = f.read()
            # lines = f.read().replace('2014/', '2014bmp/')  # coco
            lines = lines.replace("/images", "/imagesbmp")
            lines = lines.replace("/background", "/backgroundbmp")
        for ext in formats:
            lines = lines.replace(ext, ".bmp")
        with open(file.replace(".txt", "bmp.txt"), "w") as f:
            f.write(lines)


# from utils.datasets import *; recursive_dataset2bmp()
def recursive_dataset2bmp(dataset="../data/sm4_bmp"):
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + "/" + file
            s = Path(file).suffix
            if s == ".txt":  # replace text
                with open(p, "r") as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, ".bmp")
                with open(p, "w") as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, ".bmp"), cv2.imread(p))
                if s != ".bmp":
                    os.system("rm '%s'" % p)


# from utils.datasets import *; imagelist2folder()
def imagelist2folder(path="data/coco_64img.txt"):
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, "r") as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path="./new_folder"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
