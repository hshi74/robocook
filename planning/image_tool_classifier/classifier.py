import glob
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from image_tool_classifier.build_dataset import crop_image
from image_tool_classifier.dataset import ComposeTransform
from image_tool_classifier.model import resnet18
from PIL import Image
from torchvision import datasets

# from torchvision.models import resnet50
from utils.visualize import *


class ImageClassifer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.classes = sorted(args.tool_name_list)
        self.model = resnet18(num_views=4, num_classes=len(self.classes))
        # self.model = resnet50()
        # self.model.conv1 = nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # num_features = self.model.fc.in_features # extract fc layers features
        # self.model.fc = nn.Linear(num_features, len(self.classes))

        pretrained_dict = torch.load(
            self.args.tool_cls_model_path, map_location=self.device
        )
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.dim = 200
        self.new_dim = 128  # int(0.9 * self.dim)
        self.transform = ComposeTransform(
            [
                transforms.CenterCrop((self.new_dim, self.new_dim)),
                transforms.ToTensor(),
                # in + out
                # transforms.Normalize((0.5554, 0.4524, 0.4808), (0.1796, 0.1395, 0.1204)),
                # in + diff
                # transforms.Normalize((0.2780, 0.2262, 0.2400), (0.3126, 0.2538, 0.2620)),
                # in + out + nobg
                # transforms.Normalize((0.1582, 0.1208, 0.1143), (0.3403, 0.2634, 0.2499)),
                # in + out + hsv
                # transforms.Normalize((0.7380, 0.1944, 0.5519), (0.3477, 0.08418, 0.1835)),
                # in + out + hsv
                # transforms.Normalize((0.6876579, 0.18740347, 0.5375136), (0.3508519, 0.09446318, 0.18572727))
                transforms.Normalize(
                    (0.58735794, 0.21487512, 0.5874147),
                    (0.41656908, 0.1036273, 0.2237085),
                ),
            ]
        )
        self.loader = datasets.folder.default_loader
        self.used_circular_cut = 0

    def crop(self, img_dir):
        im = Image.open(img_dir)
        width, height = im.size
        if width > self.dim or height > self.dim:
            cropped_img = crop_image(img_dir, dim=self.dim, visualize=False)
            cropped_img.save(img_dir)

    def eval(self, state_cur_dict, target_shape, path=""):
        img_paths = state_cur_dict["images"] + target_shape["images"]

        img_dict = {"in": [], "out": []}
        for img_path in img_paths:
            self.crop(img_path)
            file_name = os.path.basename(img_path)
            # print(file_name)
            if "in" in file_name:
                img_dict["in"].append(self.loader(img_path).convert("HSV"))
            else:
                img_dict["out"].append(self.loader(img_path).convert("HSV"))

        imgs_in = self.transform(img_dict["in"])
        imgs_out = self.transform(img_dict["out"])

        img_list = imgs_in + imgs_out

        imgs = torch.cat(img_list, dim=0)

        visualize_tensor(imgs, path=os.path.join(path, "tensor.png"))

        imgs = imgs.to(self.device)
        output = self.model(imgs.unsqueeze(0))
        prob_output = output.softmax(dim=1).detach().cpu().numpy()[0]
        idx_pred = []
        for idx in prob_output.argsort():
            if prob_output[idx] > 0.25:
                idx_pred.insert(0, idx)
        # idx_pred = prob_output.argsort()[-1:]

        labels_pred = [self.classes[x] for x in idx_pred[:2]]

        visualize_image_pred(
            img_paths, None, idx_pred, self.classes, path=os.path.join(path, "cls.png")
        )

        if not self.used_circular_cut:
            if (
                (
                    "cutter_circular" in labels_pred
                    and not "roller_large" in labels_pred
                    and not len(labels_pred) == 1
                )
                or "pusher" in labels_pred
                or "spatula_small" in labels_pred
            ):
                print(f"{labels_pred} triggered hard code!")
                labels_pred = ["cutter_circular"]

        if "cutter_circular" in labels_pred and self.used_circular_cut > 0:
            print(f"{labels_pred} triggered hard code!")
            self.used_circular_cut = 1
            labels_pred = ["pusher"]

        if "cutter_circular" in labels_pred:
            self.used_circular_cut += 1

        # import pdb; pdb.set_trace()
        # labels_pred = ['roller_large']

        return labels_pred
