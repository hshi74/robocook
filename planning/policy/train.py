import argparse
import numpy as np
import glob
import os
import shutil
import torch

from planning.control_utils import *
from planning.policy import pointnet2_param_cls
from dataset import PolicyDataset
from datetime import datetime
from dynamics.gnn import GNN
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import *
from utils.data_utils import *
from utils.loss import *
from utils.provider import *
from utils.visualize import *

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--plan_tool", type=str, default="gripper_sym_rod")
parser.add_argument("--random_seed", type=int, default=3407)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_epoch", default=200, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--decay_rate", type=float, default=1e-4)
parser.add_argument("--use_normals", type=int, default=1)
parser.add_argument("--early_fusion", type=int, default=1)
parser.add_argument("--train_set_ratio", type=float, default=0.025)
parser.add_argument("--n_actions", type=int, default=1)
parser.add_argument("--reg_loss_type", type=str, default="smooth_l1")
parser.add_argument("--cls_weight", type=float, default=1.0)
parser.add_argument("--orient_weight", type=float, default=1.0)
parser.add_argument("--n_bin_rot", type=int, default=8)
parser.add_argument("--n_bin", type=int, default=4)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--floor_dim", type=int, default=9)
parser.add_argument("--n_particles", type=int, default=300)
parser.add_argument("--rot_aug_max", type=float, default=0.25)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_name(args):
    test_name = ["plan"]
    test_name.append(f"p={args.n_particles}")
    test_name.append(f"n={args.n_actions}")

    if args.early_fusion:
        test_name.append(f"ef=1")
    if args.use_normals:
        test_name.append(f"normal=1")

    test_name.append(f"weight={args.cls_weight}+{args.orient_weight}")
    test_name.append(f"bin={args.n_bin_rot}+{args.n_bin}")
    test_name.append(f"rot={args.rot_aug_max}")
    test_name.append(f"ratio={args.train_set_ratio}")

    if args.debug:
        test_name.append("debug")

    test_name.append(datetime.now().strftime("%b-%d-%H:%M:%S"))

    return "_".join(test_name)


def rollout(args, dy_args, gnn, planner, state_in_ori, state_out_ori):
    state_in = copy.deepcopy(state_in_ori)
    state_out = copy.deepcopy(state_out_ori)
    state_cur = copy.deepcopy(state_in_ori)

    state_in_center = torch.mean(state_in[:, :, :3], dim=1)
    state_in[:, :, :3] = state_in[:, :, :3] - state_in_center
    state_in[:, :, :3] = state_in[:, :, :3] / args.scale_ratio

    state_out_center = torch.mean(state_out[:, :, :3], dim=1)
    state_out[:, :, :3] = state_out[:, :, :3] - state_out_center
    state_out[:, :, :3] = state_out[:, :, :3] / args.scale_ratio

    points = torch.cat((state_in, state_out), dim=1)

    if not args.use_normals:
        points = points[:, :, :3]

    if args.early_fusion:
        point_labels = torch.cat(
            (
                torch.zeros((*state_in.shape[:-1], 1), device=device),
                torch.ones((*state_out.shape[:-1], 1), device=device),
            ),
            dim=1,
        )
        points = torch.cat((points, point_labels), dim=-1)

    points = points.transpose(2, 1)

    with torch.no_grad():
        pred, trans_feat = planner(points)

    pred_idx = [x[0].softmax(dim=0).argmax(dim=0).item() for x in pred[0]]
    pred_offsets = [x[0].cpu().numpy() for x in pred[1]]
    pred_params = []
    for i in range(len(pred_idx)):
        pred_param = args.bin_centers[i][pred_idx[i]] + pred_offsets[i][pred_idx[i]]
        if i % (len(pred_idx) // args.n_actions) == args.rot_idx:
            while pred_param < args.tool_params["rot_range"][0]:
                pred_param += args.tool_params["rot_scope"]
                pred_params[-1] *= (-1) ** (args.tool_params["rot_scope"] // np.pi)
            while pred_param > args.tool_params["rot_range"][1]:
                pred_param -= args.tool_params["rot_scope"]
                pred_params[-1] *= (-1) ** (args.tool_params["rot_scope"] // np.pi)
        else:
            pred_param *= args.scale_ratio
        pred_params.append(pred_param)

    pred_params_torch = torch.tensor(pred_params, dtype=torch.float32).reshape(
        (args.n_actions, -1)
    )
    center = torch.mean(state_cur.squeeze(), dim=0).cpu()
    min_bounds = torch.min(state_cur, dim=1).values.squeeze().cpu().numpy()
    init_pose_seq = params_to_init_pose(
        dy_args, center, args.tool_params, pred_params_torch
    ).numpy()
    act_seq_dense = params_to_actions(
        dy_args, args.tool_params, pred_params_torch, min_bounds, step=1
    ).numpy()
    act_seq = params_to_actions(
        dy_args, args.tool_params, pred_params_torch, min_bounds, step=dy_args.time_step
    ).numpy()

    with torch.no_grad():
        state_pred_seq, _, _ = gnn.rollout(
            copy.deepcopy(state_cur), init_pose_seq[None], act_seq_dense[None]
        )

    state_cur_wshape = np.concatenate(
        (state_cur.cpu().numpy()[0, :, :3], dy_args.floor_state, init_pose_seq[0]),
        axis=0,
    )
    state_pred_seq_wshape = add_shape_to_seq(
        dy_args, state_pred_seq.cpu().numpy()[0], init_pose_seq, act_seq
    )
    state_pred_seq_wshape = np.concatenate(
        (state_cur_wshape[None], state_pred_seq_wshape), axis=0
    )

    return pred_idx, pred_params, state_pred_seq_wshape


def test(args, dy_args, dump_path, n_rollout=(10, 2), visualize=False):
    os.system("mkdir -p " + os.path.join(dump_path, "eval"))
    tee = Tee(os.path.join(dump_path, "eval", "eval.txt"), "w")

    print("Testing...")
    phases = ["train", "valid", "test", "real"]
    # datasets = {phase: PolicyDataset(args, phase, rollout_ratio=0.005) for phase in phases}
    # dataloaders = {phase: DataLoader(datasets[phase], batch_size=1, shuffle=False,
    #     num_workers=args.num_workers) for phase in phases}

    gnn = GNN(dy_args, f"{args.plan_dataf}/dy_model.pth")

    planner = pointnet2_param_cls.get_model(args)
    planner.apply(inplace_relu)

    planner = planner.to(device)

    planner_path = os.path.join(dump_path, f"net_best.pth")
    pretrained_dict = torch.load(planner_path, map_location=device)
    planner.load_state_dict(pretrained_dict, strict=False)
    planner.eval()

    for phase in phases:
        loss_list = []
        seq_path_list = sorted(glob.glob(os.path.join(args.plan_dataf, phase, "*")))
        for i in range(len(seq_path_list)):
            if i >= n_rollout[0]:
                break
            sample_path_list = sorted(glob.glob(os.path.join(seq_path_list[i], "*")))
            for j in range(len(sample_path_list)):
                if j >= n_rollout[1]:
                    break
                state_path_list = sorted(
                    glob.glob(os.path.join(sample_path_list[j], "*.h5"))
                )
                if len(state_path_list) < args.n_actions + 1:
                    continue

                param_seq_data = np.load(
                    os.path.join(sample_path_list[j], "param_seq.npy")
                )

                state_list = []
                for k in range(len(state_path_list)):
                    state = load_data(dy_args.data_names, state_path_list[k])[0]
                    state_torch = torch.tensor(
                        state[None, : dy_args.n_particles],
                        dtype=torch.float32,
                        device=device,
                    )
                    state_list.append(state_torch)

                state_vis_dict = {"init_state": state_list[0].cpu()}
                state_pred_seq_list = []
                for k in range(len(state_list) - args.n_actions):
                    if k == 0:
                        state_in = state_list[0]
                    else:
                        state_in_normals = get_normals(state_out_pred, pkg="torch").to(
                            device
                        )
                        state_in = torch.cat((state_out_pred, state_in_normals), dim=2)

                    state_out = state_list[k + args.n_actions]

                    if state_in.shape[-1] < 6:
                        state_in_normals = get_normals(state_in, pkg="torch").to(device)
                        state_in = torch.cat((state_in, state_in_normals), dim=2)

                    if state_out.shape[-1] < 6:
                        state_out_normals = get_normals(state_out, pkg="torch").to(
                            device
                        )
                        state_out = torch.cat((state_out, state_out_normals), dim=2)

                    gt_params = param_seq_data[k : k + args.n_actions].flatten()
                    gt_ind = []
                    for ii in range(gt_params.shape[0]):
                        if ii % (gt_params.shape[0] // args.n_actions) == args.rot_idx:
                            period_ratio = np.pi * 2 / args.tool_params["rot_scope"]
                            bin_diff = -np.cos(
                                (gt_params[ii] - args.bin_centers[ii]) * period_ratio
                            )
                        else:
                            bin_diff = np.abs(
                                gt_params[ii] / args.scale_ratio - args.bin_centers[ii]
                            )
                        gt_ind.append(np.argsort(bin_diff)[0])

                    # import pdb; pdb.set_trace()
                    state_out_target = copy.deepcopy(state_out[:, :, :3])
                    pred_idx, pred_params, state_pred_seq_wshape = rollout(
                        args, dy_args, gnn, planner, state_in, state_out
                    )
                    state_out_pred = torch.tensor(
                        state_pred_seq_wshape[-1:, : dy_args.n_particles],
                        dtype=torch.float32,
                        device=device,
                    )

                    state_pred_norm = state_out_pred - torch.mean(state_out_pred, dim=1)
                    state_target_norm = state_out_target - torch.mean(
                        state_out_target, dim=1
                    )
                    chamfer_loss = chamfer(
                        state_pred_norm[0], state_target_norm[0], pkg="torch"
                    ).item()

                    state_vis_dict[
                        f"pred_{k+1}: {pred_idx}, {[round(x, 3) for x in pred_params]}"
                    ] = [state_out_pred[0].cpu()]
                    state_vis_dict[
                        f"gt_{k+1}: {gt_ind}, {[round(x, 3) for x in gt_params]}"
                    ] = [state_out_target[0].cpu()]
                    state_pred_seq_list.append(state_pred_seq_wshape)

                    if k == len(state_list) - args.n_actions - 1:
                        loss_list.append(chamfer_loss)

                if visualize:
                    render_frames(
                        dy_args,
                        list(state_vis_dict.keys()),
                        list(state_vis_dict.values()),
                        res="low",
                        axis_off=False,
                        focus=False,
                        path=os.path.join(dump_path, "eval"),
                        name=f"{phase}_{str(i).zfill(3)}_{str(j).zfill(3)}.png",
                    )

                    render_anim(
                        dy_args,
                        [f"GNN"],
                        [np.concatenate(state_pred_seq_list, axis=0)],
                        res="low",
                        path=os.path.join(
                            dump_path,
                            "eval",
                            f"{phase}_{str(i).zfill(3)}_{str(j).zfill(3)}.mp4",
                        ),
                    )

        loss_avg = np.mean(np.stack(loss_list), axis=0)
        print(f"{phase} phase average l1 loss for each param: {loss_avg}")


def compute_stats(args):
    seq_path_list = sorted(glob.glob(os.path.join(args.plan_dataf, "train", "*")))
    param_seq_list = []
    for i in range(len(seq_path_list)):
        sample_path_list = sorted(glob.glob(os.path.join(seq_path_list[i], "*")))
        for j in range(len(sample_path_list)):
            if j >= args.train_set_ratio * len(sample_path_list):
                break
            state_path_list = glob.glob(os.path.join(sample_path_list[j], "*.h5"))
            if len(state_path_list) < args.n_actions + 1:
                continue
            param_seq_data = np.load(os.path.join(sample_path_list[j], "param_seq.npy"))
            for k in range(len(state_path_list) - args.n_actions):
                params = param_seq_data[k : k + args.n_actions].flatten()
                for ii in range(params.shape[0]):
                    if ii % (params.shape[0] // args.n_actions) != args.rot_idx:
                        params[ii] /= args.scale_ratio
                param_seq_list.append(params)

    param_seq = np.stack(param_seq_list)
    mean = np.mean(param_seq, axis=0)
    std = np.std(param_seq, axis=0)
    param_seq_norm = param_seq  # (param_seq - mean) / std
    param_seq_min = np.min(param_seq_norm, axis=0)
    param_seq_max = np.max(param_seq_norm, axis=0)
    bin_centers = []
    for i in range(param_seq.shape[1]):
        if i % (param_seq.shape[1] // args.n_actions) == args.rot_idx:
            half_bin_size = (
                args.tool_params["rot_range"][1] - args.tool_params["rot_range"][0]
            ) / args.n_bins[i]
            bin_centers.append(
                args.tool_params["rot_range"][0]
                + half_bin_size * np.arange(args.n_bins[i])
            )
        else:
            half_bin_size = (param_seq_max[i] - param_seq_min[i]) / (args.n_bins[i] - 1)
            bin_centers.append(
                param_seq_min[i] + half_bin_size * np.arange(args.n_bins[i])
            )

    print(
        f"mean: {mean}\nstd: {std}\nparam_seq_min: {param_seq_min}\nparam_seq_max: {param_seq_max}"
    )
    print(f"bin_centers: {bin_centers}")

    return bin_centers


def train(args, dump_path):
    tee = Tee(os.path.join(dump_path, "train.txt"), "w")

    phases = {"train": args.train_set_ratio, "valid": args.train_set_ratio}
    datasets = {
        phase: PolicyDataset(args, phase, rollout_ratio=r)
        for phase, r in phases.items()
    }

    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=args.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=args.num_workers,
        )
        for phase in phases.keys()
    }

    planner = pointnet2_param_cls.get_model(args)
    criterion = pointnet2_param_cls.get_loss(args)
    planner.apply(inplace_relu)

    planner = planner.to(device)
    criterion = criterion.to(device)

    # planner_path = os.path.join(dump_path, f'net_best.pth')
    # pretrained_dict = torch.load(planner_path, map_location=device)
    # planner.load_state_dict(pretrained_dict, strict=False)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            planner.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(planner.parameters(), lr=0.01, momentum=0.9)

    # scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.8, patience=3, verbose=True
    )
    best_valid_loss, best_epoch = np.inf, 0

    train_stats = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_reg_loss": [],
        "valid_loss": [],
        "valid_cls_loss": [],
        "valid_reg_loss": [],
    }

    print("Training...")
    for epoch in range(args.n_epoch):
        for phase in phases:
            with torch.set_grad_enabled(phase == "train"):
                planner = planner.train(phase == "train")
                cls_loss_list = []
                reg_loss_list = []
                loss_list = []
                for i, (points, target_idx, target_params) in enumerate(
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
                            points[:, : args.n_particles], max_dropout_ratio=0.4
                        )
                        points[:, args.n_particles :] = random_point_dropout(
                            points[:, args.n_particles :], max_dropout_ratio=0.4
                        )

                        # points[:, :, :3] = add_gaussian_noise(points[:, :, :3])

                        target_idx = target_idx.numpy()
                        target_params = target_params.numpy()

                        (
                            points,
                            target_params,
                            target_idx,
                        ) = random_rotate_z_point_cloud_with_normal(
                            args, points, target_params, target_idx
                        )

                        points[:, : args.n_particles, :3] = shift_point_cloud(
                            points[:, : args.n_particles, :3]
                        )
                        points[:, args.n_particles :, :3] = shift_point_cloud(
                            points[:, args.n_particles :, :3]
                        )

                        # import pdb; pdb.set_trace()
                        # target_params_denorm = target_params[0] * args.stats['std'] + args.stats['mean']
                        # render_frames(args, [f'IN: {list(target_idx[0]), list(target_params_denorm)}', 'OUT'],
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
                        target_idx = torch.tensor(target_idx, dtype=torch.float32)
                        target_params = torch.tensor(target_params, dtype=torch.float32)

                    points = points.to(device=device, dtype=torch.float32)
                    points = points.transpose(2, 1)
                    target_idx = target_idx.to(device)
                    target_params = target_params.to(device)

                    # import pdb; pdb.set_trace()
                    pred, trans_feat = planner(points)

                    cls_loss, reg_loss = criterion(
                        pred, (target_idx.long(), target_params)
                    )
                    loss = cls_loss + reg_loss

                    cls_loss_list.append(cls_loss)
                    reg_loss_list.append(reg_loss)
                    loss_list.append(loss)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                cls_loss_avg = torch.mean(torch.stack(cls_loss_list))
                reg_loss_avg = torch.mean(torch.stack(reg_loss_list))
                loss_avg = torch.mean(torch.stack(loss_list))
                print(
                    f"lr: {get_lr(optimizer):.6f}; {phase} loss: {loss_avg.item():.6f}; "
                    + f"cls loss: {cls_loss_avg.item():.6f}; reg loss: {reg_loss_avg.item():.6f}"
                )
                train_stats[f"{phase}_cls_loss"].append(cls_loss_avg.item())
                train_stats[f"{phase}_reg_loss"].append(reg_loss_avg.item())
                train_stats[f"{phase}_loss"].append(loss_avg.item())

                if phase == "valid":
                    scheduler.step(loss_avg)
                    if loss_avg.item() < best_valid_loss:
                        best_valid_loss = loss_avg.item()
                        best_epoch = epoch
                        best_model_path = f"{dump_path}/net_best.pth"
                        torch.save(planner.state_dict(), best_model_path)
                        plateau_epoch = 0
                    else:
                        plateau_epoch += 1

        if plateau_epoch >= 10:
            print(f"Breaks after not improving for {plateau_epoch} epoches!")
            break

    print(f"Best epoch {best_epoch}: best loss: {best_valid_loss:.6f}!")

    plot_train_loss(train_stats, path=os.path.join(dump_path, "training_loss.png"))


def main():
    args = parser.parse_args()
    set_seed(args.random_seed)
    args.plan_dataf = f"data/planning_data/data_{args.plan_tool}_robot_v4_surf_nocorr_full_normal_keyframe=16_action=2_p={args.n_particles}"
    args.device = device

    dump_path = os.path.join(args.plan_dataf, "dump", get_test_name(args))
    # dump_path = os.path.join(args.plan_dataf, 'dump', 'plan_p=300_n=2_ef=1_normal=1_weight=1.0+1.0_bin=32+8_rot=0.25_ratio=1.0_Nov-22-14:21:43')
    os.system("mkdir -p " + dump_path)

    dy_args_dict = np.load(f"{args.plan_dataf}/args.npy", allow_pickle=True).item()
    dy_args = copy.deepcopy(args)
    dy_args.__dict__ = dy_args_dict
    dy_args.ee_fingertip_T_mat = np.array(
        [[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]]
    )
    dy_args.env = args.plan_tool

    with open("config/tool_plan_params.yml", "r") as f:
        args.tool_params = yaml.load(f, Loader=yaml.FullLoader)[args.plan_tool]

    tool_action_space_size = {
        "gripper_asym": 3,
        "gripper_sym_rod": 3,
        "gripper_sym_plane": 3,
        "roller_small": 4,
        "roller_large": 4,
        "press_square": 4,
        "press_circle": 3,
        "punch_square": 4,
        "punch_circle": 3,
    }

    n_bins = []
    n_params = tool_action_space_size[args.plan_tool] * args.n_actions
    args.rot_idx = 1 if "gripper" in args.plan_tool else 3
    for i in range(n_params):
        if i % tool_action_space_size[args.plan_tool] == args.rot_idx:
            # periodic
            n_bins.append(args.n_bin_rot)
        else:
            n_bins.append(args.n_bin + 1)
    args.n_bins = np.array(n_bins)

    args.rot_aug_max *= np.pi
    args.scale_ratio = 0.035

    args.bin_centers = compute_stats(args)
    args.bin_centers_torch = [
        torch.tensor(x, dtype=torch.float32, device=device) for x in args.bin_centers
    ]

    if args.debug:
        args.n_epoch = 1
        args.train_set_ratio = 0.005
        args.batch_size = 2

    shutil.copy("utils/provider.py", dump_path)
    shutil.copy("planning/policy/dataset.py", dump_path)
    shutil.copy("planning/policy/pointnet2_param_cls.py", dump_path)
    shutil.copy("planning/policy/train.py", dump_path)

    train(args, dump_path)
    test(args, dy_args, dump_path, visualize=True)

    with open(os.path.join(dump_path, "args.npy"), "wb") as f:
        np.save(f, args.__dict__)


if __name__ == "__main__":
    main()
