import argparse
import numpy as np
import os
import shutil
import sys
import torch

from planning.pcd_tool_classifier import pointnet2_tool_cls
from dataset import ToolDataset
from datetime import datetime
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import gen_args
from utils.data_utils import *
from utils.provider import *
from utils.visualize import *


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--tool_dataf", type=str, default="data/tool_classification")
parser.add_argument("--random_seed", type=int, default=3407)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_epoch", default=200, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--decay_rate", type=float, default=1e-4)
parser.add_argument("--n_particles", type=int, default=4096)
parser.add_argument("--n_class", default=15, type=int)
parser.add_argument("--use_normals", type=int, default=0)
parser.add_argument("--use_rgb", type=int, default=1)
parser.add_argument("--early_fusion", type=int, default=1)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--floor_dim", type=int, default=9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_name(args):
    test_name = ["cls"]
    test_name.append(f"p={args.n_particles}")

    if args.early_fusion:
        test_name.append(f"ef=1")
    if args.use_rgb:
        test_name.append(f"rgb=1")

    if args.debug:
        test_name.append("debug")

    test_name.append(datetime.now().strftime("%b-%d-%H:%M:%S"))

    return "_".join(test_name)


def test(args, dump_path, visualize=False):
    eval_path = os.path.join(dump_path, "eval")
    os.system("rm -r " + eval_path)
    os.system("mkdir -p " + eval_path)

    tee = Tee(os.path.join(eval_path, "eval.txt"), "w")

    classifier = pointnet2_tool_cls.get_model(args)
    criterion = pointnet2_tool_cls.get_loss(args)
    classifier.apply(inplace_relu)

    classifier_path = os.path.join(dump_path, f"net_best.pth")
    pretrained_dict = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(pretrained_dict, strict=False)
    classifier.eval()

    classifier = classifier.to(device)
    criterion = criterion.to(device)

    print("Testing...")
    for phase in ["train", "valid", "test"]:
        test_set = ToolDataset(args, phase)
        dataloader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=args.num_workers
        )

        mean_correct = []
        class_acc = np.zeros((args.n_class, 3))

        idx_preds = []
        for i, (points, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
            points = points.to(device=device, dtype=torch.float32)
            points = points.transpose(2, 1)
            target = target.to(device=device)

            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            idx_preds.append(pred_choice.item())

            for cat in np.unique(target.cpu()):
                classacc = (
                    pred_choice[target == cat]
                    .eq(target[target == cat].long().data)
                    .cpu()
                    .sum()
                )
                class_acc[cat, 0] += classacc.item() / float(
                    points[target == cat].size()[0]
                )
                class_acc[cat, 1] += 1

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            if visualize and not correct:
                state = points.transpose(2, 1)
                state_in = state[0, : args.n_particles]
                state_out = state[0, args.n_particles :]
                print(f"False prediction!")
                # render_frames(args, [f'Pred: {test_set.classes[pred_choice.item()]}', f'Label: {test_set.classes[target.item()]}'],
                #     [state_in.cpu(), state_out.cpu()], res='low', axis_off=False, focus=False, path=eval_path,
                #     name=f'{phase}_{str(i).zfill(3)}.png')
                visualize_pcd_pred(
                    [
                        f"Pred: {test_set.classes[pred_choice.item()]}",
                        f"Label: {test_set.classes[target.item()]}",
                    ],
                    [state_in.cpu(), state_out.cpu()],
                    path=os.path.join(eval_path, f"{phase}_{str(i).zfill(3)}.png"),
                )

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        print(f"instance accuracy: {instance_acc:.3f}; class accuracy: {class_acc:.3f}")

        if visualize:
            idx_labels = list(list(zip(*test_set.point_set))[1])
            plot_cm(
                test_set,
                idx_labels,
                idx_preds,
                path=os.path.join(eval_path, f"cm_{phase}.png"),
            )

    return instance_acc, class_acc


def train(args, dump_path):
    tee = Tee(os.path.join(dump_path, "train.txt"), "w")

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

    classifier = pointnet2_tool_cls.get_model(args)
    criterion = pointnet2_tool_cls.get_loss(args)
    classifier.apply(inplace_relu)

    classifier = classifier.to(device)
    criterion = criterion.to(device)

    # classifier_path = os.path.join(dump_path, f"net_best.pth")
    # pretrained_dict = torch.load(classifier_path, map_location=device)
    # classifier.load_state_dict(pretrained_dict, strict=False)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.8, patience=3, verbose=True
    )
    best_epoch, best_valid_loss, best_instance_acc = 0, np.inf, 0

    train_stats = {
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_loss": [],
        "valid_loss": [],
    }

    print("Training...")
    for epoch in range(args.n_epoch):
        for phase in phases:
            with torch.set_grad_enabled(phase == "train"):
                classifier = classifier.train(phase == "train")
                mean_correct = []
                loss_list = []
                for i, (points, target) in enumerate(
                    tqdm(dataloaders[phase], desc=f"epoch {epoch}/{args.n_epoch}")
                ):
                    # import pdb; pdb.set_trace()
                    if phase == "train":
                        # data augmentation
                        points = points.numpy()

                        points[:, : args.n_particles] = shuffle_points(
                            points[:, : args.n_particles]
                        )
                        points[:, args.n_particles :] = shuffle_points(
                            points[:, args.n_particles :]
                        )
                        points[:, : args.n_particles] = random_point_dropout(
                            points[:, : args.n_particles]
                        )
                        points[:, args.n_particles :] = random_point_dropout(
                            points[:, args.n_particles :]
                        )

                        # points[:, :, :3] = add_gaussian_noise(points[:, :, :3], noise_scale=0.01)

                        # points, _, _ = random_rotate_z_point_cloud_with_normal(args, points, None, None)

                        points[:, :, :3] = random_scale_point_cloud(points[:, :, :3])
                        points[:, :, :3] = rotate_point_cloud(points[:, :, :3])
                        points[:, :, :3] = shift_point_cloud(points[:, :, :3])

                        # import pdb; pdb.set_trace()
                        # target_params_denorm = target_params[0] * args.stats['std'] + args.stats['mean']
                        # render_frames(args, ['IN', 'OUT'],
                        #     [points[:1, :args.n_particles, :3], points[:1, args.n_particles:, :3]], res='low', axis_off=False, focus=False)

                        # dy_args_dict = np.load(f'{args.plan_dataf}/args.npy', allow_pickle=True).item()
                        # dy_args = copy.deepcopy(args)
                        # dy_args.__dict__ = dy_args_dict
                        # dy_args.floor_unit_size = 0.05
                        # dy_args.ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])

                        # init_pose = params_to_init_pose(dy_args, np.zeros(3), tool_params, torch.tensor(target_params_denorm[None]))[0]
                        # render_frames(dy_args, [f'IN: {list(target_idx[0]), list(target_params_denorm)}', 'OUT'],
                        #     [np.concatenate((points[:1, :args.n_particles, :3], init_pose[None]), axis=1),
                        #     np.concatenate((points[:1, args.n_particles:, :3], init_pose[None]), axis=1)], res='low', axis_off=False, focus=True)

                        points = torch.tensor(points, dtype=torch.float32)

                    points = points.to(device=device, dtype=torch.float32)
                    points = points.transpose(2, 1)
                    target = target.to(device)

                    # import pdb; pdb.set_trace()
                    pred, trans_feat = classifier(points)
                    loss = criterion(pred, target.long())
                    pred_choice = pred.data.max(1)[1]

                    correct = pred_choice.eq(target.long().data).cpu().sum()
                    mean_correct.append(correct.item() / float(points.size()[0]))
                    loss_list.append(loss)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            instance_acc = np.mean(mean_correct)
            loss_avg = torch.mean(torch.stack(loss_list))
            print(
                f"lr: {get_lr(optimizer):.6f}; {phase} loss: {loss_avg.item():.6f}; "
                + f"instance accuracy: {instance_acc:.6f}"
            )
            train_stats[f"{phase}_accuracy"].append(instance_acc)
            train_stats[f"{phase}_loss"].append(loss_avg.item())

            if phase == "valid":
                scheduler.step(loss_avg)
                if loss_avg < best_valid_loss:
                    best_epoch = epoch
                    best_valid_loss = loss_avg
                    best_instance_acc = instance_acc
                    best_model_path = f"{dump_path}/net_best.pth"
                    torch.save(classifier.state_dict(), best_model_path)
                    plateau_epoch = 0
                else:
                    plateau_epoch += 1

        if plateau_epoch >= 10:
            print(f"Breaks after not improving for {plateau_epoch} epoches!")
            break

    print(
        f"Best epoch {best_epoch}: loss: {best_valid_loss:.6f}; instance accuracy: {best_instance_acc:.6f}!"
    )

    plot_train_loss(train_stats, path=os.path.join(dump_path, "training_loss.png"))


def main():
    args = parser.parse_args()
    set_seed(args.random_seed)

    dump_path = os.path.join("dump", "tool_classification", get_test_name(args))
    os.system("mkdir -p " + dump_path)

    args.axes = np.array([0, 1, 2])
    args.tool_type = "gripper_sym_rod_robot_v1"

    if args.debug:
        args.n_epoch = 1

    shutil.copy("utils/provider.py", dump_path)
    shutil.copy("planning/pcd_tool_classifier/dataset.py", dump_path)
    shutil.copy("planning/pcd_tool_classifier/pointnet2_tool_cls.py", dump_path)
    shutil.copy("planning/pcd_tool_classifier/train.py", dump_path)

    train(args, dump_path)
    test(args, dump_path, visualize=True)


if __name__ == "__main__":
    main()
