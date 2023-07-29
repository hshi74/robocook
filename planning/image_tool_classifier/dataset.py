import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.visualize import visualize_tensor


def plot(imgs, row_title=None):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

    plt.show()


class ComposeTransform(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            for i in range(len(imgs)):
                # state = torch.get_rng_state()
                imgs[i] = t(imgs[i])
                # torch.set_rng_state(state)
        return imgs


class ToolDataset(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(args.tool_dataf, phase)
        self.dim = 200
        self.new_dim = 128  # int(0.9 * self.dim)
        self.train_transform = ComposeTransform(
            [
                # transforms.CenterCrop((self.dim, self.dim)),
                transforms.ColorJitter(
                    brightness=0.3
                ),  # , contrast=0.3, saturation=0.3),
                # transforms.RandomCrop(size=int(200*0.9), padding=int(200*0.1)),
                # transforms.RandomHorizontalFlip(p=0.1),
                # transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(
                    15, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.RandomCrop(size=self.new_dim),
                transforms.ToTensor(),
                # transforms.Normalize((0.5554, 0.4524, 0.4808), (0.1796, 0.1395, 0.1204)),
                # transforms.Normalize((0.6876579, 0.18740347, 0.5375136), (0.3508519, 0.09446318, 0.18572727))
                transforms.Normalize(
                    (0.58735794, 0.21487512, 0.5874147),
                    (0.41656908, 0.1036273, 0.2237085),
                ),
            ]
        )

        self.test_transform = ComposeTransform(
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
        self.classes, self.class_to_idx = self._get_classes(self.data_dir)
        self.samples = self._get_ims(self.data_dir)
        self.loader = datasets.folder.default_loader

    def _get_classes(self, root):
        classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        class_to_idx = {class_name: x for x, class_name in enumerate(classes)}
        return classes, class_to_idx

    def _get_ims(self, root):
        im_set = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_idx = self.class_to_idx[target_class]
            class_dir = os.path.join(root, target_class)
            for (
                class_dir,
                dirs,
                files,
            ) in os.walk(class_dir, topdown=True):
                counter = 0
                view_path = []
                for fname in sorted(files):
                    if ".txt" in fname:
                        continue
                    cam_idx = int(fname.split(".")[0].split("_")[-1])
                    if cam_idx in list(range(1, self.args.num_views + 1)):
                        view_path.append(os.path.join(class_dir, fname))
                        if (
                            counter % (self.args.num_views * 2)
                            == self.args.num_views * 2 - 1
                        ):
                            im_set.append((view_path, class_idx))
                            view_path = []
                        counter += 1

        return im_set

    def __getitem__(self, index):
        img_paths, target = self.samples[index]
        # print(img_paths)
        img_dict = {"in": [], "out": []}
        for img_path in img_paths:
            file_name = os.path.basename(img_path)
            # print(file_name)
            if "in" in file_name:
                img_dict["in"].append(self.loader(img_path).convert("HSV"))
            else:
                img_dict["out"].append(self.loader(img_path).convert("HSV"))

        # plot(img_dict['in'] + img_dict['out'])

        if self.phase == "train":
            imgs_in = self.train_transform(img_dict["in"])
            imgs_out = self.train_transform(img_dict["out"])
        else:
            imgs_in = self.test_transform(img_dict["in"])
            imgs_out = self.test_transform(img_dict["out"])

        # imgs_out = self.test_transform(img_dict['out'])

        img_list = imgs_in + imgs_out

        imgs = torch.cat(img_list, dim=0)

        # visualize_tensor(imgs, mode='HSV')

        return imgs, target

    def __len__(self):
        return len(self.samples)
