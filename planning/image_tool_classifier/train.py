import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn

from planning.image_tool_classifier.model import resnet18
from planning.image_tool_classifier.dataset import ToolDataset
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# from torchvision.models import resnet18, resnet50, ResNet18_Weights
from tqdm import tqdm
from utils.data_utils import set_seed, Tee
from utils.visualize import *


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--tool_dataf",
    type=str,
    default="data/image_classifier/data_classifier_08-24_epwise_final_v4",
)
parser.add_argument("--random_seed", type=int, default=3407)
parser.add_argument("--num_views", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.0004)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=10)

args = parser.parse_args()
set_seed(args.random_seed)

# dump_path = os.path.join(args.tool_dataf, 'dump', datetime.now().strftime("%b-%d-%H:%M:%S"))
dump_path = os.path.join(args.tool_dataf, "dump", "Sep-27-00:10:07")

os.system("mkdir -p " + dump_path)

tee = Tee(os.path.join(dump_path, "train.log"), "w")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(visualize=False):
    for dataset in ["train", "valid", "test"]:
        os.system("mkdir -p " + os.path.join(dump_path, dataset, "images", "true"))
        os.system("mkdir -p " + os.path.join(dump_path, dataset, "images", "false"))

        test_set = ToolDataset(args, dataset)
        test_set.phase = "test"
        dataloader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # model = resnet50()
        # model.conv1 = nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # num_features = model.fc.in_features # extract fc layers features
        # model.fc = nn.Linear(num_features, len(test_set.classes))
        model = resnet18(num_views=args.num_views, num_classes=len(test_set.classes))
        model_path = os.path.join(dump_path, f"net_best_v11.pth")
        pretrained_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        running_loss = 0.0
        succ_hit = 0
        idx_preds = []
        for i, data in enumerate(tqdm(dataloader)):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = criterion(output, label)
            pred = output.softmax(dim=1).argmax(dim=1)
            idx_preds.append(pred.cpu().numpy()[0])
            is_succ = torch.sum(pred == label)
            if visualize:
                img_paths, target = test_set.samples[i]
                if is_succ == 0:
                    print(f"False prediction!")
                    vis_path = os.path.join(
                        dump_path, dataset, "images", "false", f"{str(i).zfill(3)}.png"
                    )
                    visualize_image_pred(
                        img_paths,
                        target,
                        pred.cpu().numpy(),
                        test_set.classes,
                        path=vis_path,
                    )
                # else:
                #     vis_path = os.path.join(dump_path, dataset, 'images', 'true', f'{str(i).zfill(3)}.png')

            succ_hit += is_succ

            running_loss += loss.item()

        print(
            f"test loss: {running_loss / len(test_set):.3f}; success: {succ_hit / (len(test_set))}"
        )

        if visualize:
            idx_labels = list(list(zip(*test_set.samples))[1])
            plot_cm(
                test_set,
                idx_labels,
                idx_preds,
                path=os.path.join(dump_path, dataset, "cm.png"),
            )


def train():
    phases = ["train", "valid"]

    datasets = {phase: ToolDataset(args, phase) for phase in phases}

    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=args.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=args.num_workers,
        )
        for phase in phases
    }

    # model = resnet50()
    # model.conv1 = nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # num_features = model.fc.in_features # extract fc layers features
    # model.fc = nn.Linear(num_features, len(datasets['train'].classes))
    model = resnet18(
        num_views=args.num_views, num_classes=len(datasets["train"].classes)
    )
    model_path = os.path.join(dump_path, f"net_best_v11.pth")
    pretrained_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_dict, strict=False)

    params = model.parameters()
    model = model.to(device)

    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.8, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_epoch, best_valid_loss, best_valid_accuracy = 0, np.inf, 0
    plateau_epoch = 0

    train_stats = {
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_loss": [],
        "valid_loss": [],
    }

    for epoch in range(args.n_epoch):
        rgb_mean_list = []
        rgb_std_list = []
        for phase in phases:
            running_loss = 0.0
            succ_hit = 0
            model.train(phase == "train")
            for i, data in enumerate(
                tqdm(dataloaders[phase], desc=f"epoch {epoch}/{args.n_epoch}")
            ):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                output = model(img)

                loss = criterion(output, label)
                succ_hit += torch.sum(
                    output.softmax(dim=1).argmax(dim=1) == label
                ).item()

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # rgb_mean = torch.stack([
                    #     torch.mean(torch.mean(img[:, ::3], axis=(0, 2, 3)).reshape((4, 2)), axis=1),
                    #     torch.mean(torch.mean(img[:, 1::3], axis=(0, 2, 3)).reshape((4, 2)), axis=1),
                    #     torch.mean(torch.mean(img[:, 2::3], axis=(0, 2, 3)).reshape((4, 2)), axis=1),
                    # ]).T
                    # rgb_std = torch.stack([
                    #     torch.mean(torch.std(img[:, ::3], axis=(0, 2, 3)).reshape((4, 2)), axis=1),
                    #     torch.mean(torch.std(img[:, 1::3], axis=(0, 2, 3)).reshape((4, 2)), axis=1),
                    #     torch.mean(torch.std(img[:, 2::3], axis=(0, 2, 3)).reshape((4, 2)), axis=1),
                    # ]).T

                    rgb_mean = np.array(
                        [
                            torch.mean(img[:, ::3]).cpu(),
                            torch.mean(img[:, 1::3]).cpu(),
                            torch.mean(img[:, 2::3]).cpu(),
                        ]
                    )
                    rgb_std = np.array(
                        [
                            torch.std(img[:, ::3]).cpu(),
                            torch.std(img[:, 1::3].cpu()),
                            torch.std(img[:, 2::3]).cpu(),
                        ]
                    )
                    rgb_mean_list.append(rgb_mean)
                    rgb_std_list.append(rgb_std)

                running_loss += loss.item()

            loss_avg = running_loss / len(datasets[phase])
            accuracy = succ_hit / len(datasets[phase])
            print(
                f"[{epoch}] {phase} loss: {loss_avg:.6f} "
                + f"{phase} success {accuracy:.6f}"
            )
            train_stats[f"{phase}_loss"].append(min(loss_avg, 1))
            train_stats[f"{phase}_accuracy"].append(accuracy)
            stats_mean = np.mean(np.stack(rgb_mean_list), axis=0)
            stats_std = np.mean(np.stack(rgb_std_list), axis=0)
            print(f"{stats_mean} +- {stats_std}")

            if phase == "valid":
                scheduler.step()
                if loss_avg < best_valid_loss:
                    best_epoch = epoch
                    best_valid_loss = loss_avg
                    best_valid_accuracy = accuracy
                    best_model_path = f"{dump_path}/net_best_v11.pth"
                    torch.save(model.state_dict(), best_model_path)
                    plateau_epoch = 0
                else:
                    plateau_epoch += 1

        if plateau_epoch >= 10:
            print(f"Breaks after not improving for {plateau_epoch} epoches!")
            break

    print(
        f"Best epoch {best_epoch}: valid loss: {best_valid_loss:.6f} valid accuracy: {best_valid_accuracy:.6f}!"
    )
    plot_train_loss(train_stats, path=os.path.join(dump_path, "training_loss.png"))


def main():
    train()
    test(visualize=True)


if __name__ == "__main__":
    main()
