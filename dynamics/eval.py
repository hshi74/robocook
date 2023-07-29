import glob
import numpy as np
import os
import torch

from dynamics.gnn import GNN
from perception.pcd_utils import alpha_shape_mesh_reconstruct
from tqdm import tqdm
from utils.config import *
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


def evaluate(args, load_args=False):
    if not "dump" in args.dy_model_path:
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        args.dy_model_path = os.path.join(
            cd, "..", "dump", "dynamics", f"dump_{args.tool_type}", args.dy_model_path
        )

    args.dy_out_path = os.path.dirname(args.dy_model_path)

    args_path = os.path.join(args.dy_out_path, "args.npy")
    if load_args and os.path.exists(args_path):
        args.__dict__ = np.load(args_path, allow_pickle=True).item()

    print_args(args)

    dy_eval_path = os.path.join(args.dy_out_path, "eval")
    dy_plot_path = os.path.join(dy_eval_path, "plot")
    dy_anim_path = os.path.join(dy_eval_path, "anim")
    os.system("mkdir -p " + dy_plot_path)
    os.system("mkdir -p " + dy_anim_path)

    tee = Tee(os.path.join(dy_eval_path, "eval.txt"), "w")

    train_stats_path = os.path.join(args.dy_out_path, "train_stats.npy")
    if os.path.exists(train_stats_path):
        with open(train_stats_path, "rb") as f:
            train_stats = np.load(f, allow_pickle=True)
            train_stats = train_stats[None][0]

            plot_train_loss(
                train_stats, path=os.path.join(dy_eval_path, "training_loss.png")
            )

    if args.env == "all":
        load_tool_info = True
    else:
        load_tool_info = False

    gnn = GNN(args, args.dy_model_path)

    for dataset in ["test", "valid", "train"]:
        loss_dict_all = {
            "chamfer": {"surf": [], "full": [], "synth": []},
            "emd": {"surf": [], "full": [], "synth": []},
            "h": {"surf": [], "full": [], "synth": []},
            "iou": {"full": []},
        }
        seq_len_max = 0

        dataset_path = os.path.join(args.dy_data_path, dataset)
        if load_tool_info:
            tool_info_dict = np.load(
                os.path.join(dataset_path, "tool_info.npy"), allow_pickle=True
            )

        dataset_size = len(glob.glob(os.path.join(dataset_path, "*")))
        print(f"Rolling out on the {dataset} set:")
        for idx in tqdm(range(min(args.n_rollout, dataset_size))):
            vid_path = os.path.join(dataset_path, str(idx).zfill(3))
            if not os.path.exists(vid_path):
                continue

            if load_tool_info:
                for tool, vid_list in tool_info_dict.item().items():
                    if idx in vid_list:
                        args.tool_type = tool
                        args.env = args.tool_type.split("_robot")[0]
                        break

            if args.data_type == "synthetic":
                gt_vid_path = vid_path.replace("synthetic", "gt").replace(
                    f"_time_step={args.data_time_step}", ""
                )
            else:
                gt_vid_path = vid_path

            # load data
            n_frames = len(glob.glob(os.path.join(gt_vid_path, "*.h5")))
            state_gt_seq_dense = []
            for step in range(0, n_frames, args.data_time_step):
                frame_name = str(step).zfill(3) + ".h5"
                gt_data = load_data(
                    args.data_names, os.path.join(gt_vid_path, frame_name)
                )[0]
                state_gt_seq_dense.append(gt_data)
            state_gt_seq_dense = np.stack(state_gt_seq_dense)
            act_seq_dense = get_act_seq_from_state_seq(
                args, state_gt_seq_dense[:, :, :3]
            )

            state_gt_seq = []
            frame_start = (
                state_gt_seq_dense.shape[0]
                - 1
                - (state_gt_seq_dense.shape[0] - 1) // args.time_step * args.time_step
            )
            frame_list = list(
                range(frame_start, state_gt_seq_dense.shape[0], args.time_step)
            )
            # print(frame_start,  state_gt_seq_dense.shape[0], args.time_step)
            # print(frame_list)
            state_gt_seq = state_gt_seq_dense[frame_list]

            state_seq = []
            if args.data_type == "synthetic":
                vid_path = os.path.join(vid_path, "000")
                for step in frame_list:
                    frame_name = str(step).zfill(3) + ".h5"
                    data = load_data(
                        args.data_names, os.path.join(vid_path, frame_name)
                    )
                    state_seq.append(data[0])
                state_seq = np.stack(state_seq)
            else:
                state_seq = state_gt_seq

            if args.surface_sample:
                state_surf_seq = state_seq
            else:
                surf_data_path = f"{args.dy_data_path}_surf_nocorr"
                if args.full_repr:
                    surf_data_path += "_full"

                state_surf_seq = []
                for step in frame_list:
                    frame_name = str(step).zfill(3) + ".h5"
                    surf_data = load_data(
                        args.data_names,
                        os.path.join(
                            surf_data_path, dataset, str(idx).zfill(3), frame_name
                        ),
                    )
                    state_surf_seq.append(surf_data[0])
                state_surf_seq = np.stack(state_surf_seq)

            # init_pose_seq: (n_moves, n_shapes, 3)
            # act_seq: (n_moves, n_steps, 6 * n_tools)
            init_pose_seq = np.expand_dims(
                copy.deepcopy(state_gt_seq[0, args.n_particles + args.floor_dim :, :3]),
                0,
            )
            act_seq = get_act_seq_from_state_seq(args, state_gt_seq[:, :, :3])

            # import pdb; pdb.set_trace()
            with torch.no_grad():
                state_pred_seq, attn_mask_pred, rels_pred = gnn.rollout(
                    copy.deepcopy(state_gt_seq[0]),
                    np.expand_dims(init_pose_seq, 0),
                    np.expand_dims(act_seq_dense, 0),
                )

            state_pred_seq = add_shape_to_seq(
                args, state_pred_seq.cpu().numpy()[0], init_pose_seq, act_seq
            )
            state_pred_seq = np.concatenate(
                (copy.deepcopy(state_gt_seq[:1, :, :3]), state_pred_seq)
            )

            if args.surface_sample:
                state_pred_surf_seq = state_pred_seq
            else:
                state_pred_surf_seq = []
                for i in range(state_pred_seq.shape[0]):
                    state_pred = state_pred_seq[i][: args.n_particles]
                    state_pred_upsample = upsample(state_pred)
                    state_pred_pcd = o3d.geometry.PointCloud()
                    state_pred_pcd.points = o3d.utility.Vector3dVector(
                        state_pred_upsample
                    )
                    surf_mesh = alpha_shape_mesh_reconstruct(state_pred_pcd, alpha=0.01)
                    state_pred_pcd_surf = (
                        o3d.geometry.TriangleMesh.sample_points_poisson_disk(
                            surf_mesh, args.n_particles
                        )
                    )
                    state_pred_surf = np.asarray(state_pred_pcd_surf.points)
                    state_pred_surf_seq.append(state_pred_surf)

                state_pred_surf_seq = np.concatenate(
                    (
                        np.array(state_pred_surf_seq),
                        state_pred_seq[:, args.n_particles :],
                    ),
                    axis=1,
                )

            attn_mask_pred = np.squeeze(attn_mask_pred.cpu().numpy()[0])
            attn_mask_pred = np.concatenate(
                (np.zeros((args.n_his, args.n_particles)), attn_mask_pred), axis=0
            )

            rels_pred = rels_pred[0]
            max_n_rel = max([rels.shape[0] for rels in rels_pred])
            for i in range(len(rels_pred)):
                rels_pred[i] = np.concatenate(
                    (
                        np.zeros(
                            (max_n_rel - rels_pred[i].shape[0], rels_pred[i].shape[1]),
                            dtype=np.int16,
                        ),
                        rels_pred[i],
                    ),
                    axis=0,
                )

            rels_pred = np.stack(rels_pred)
            rels_pred = np.concatenate(
                (
                    rels_pred,
                    np.zeros(
                        (args.n_his, max_n_rel, rels_pred[0].shape[1]), dtype=np.int16
                    ),
                ),
                axis=0,
            )

            seq_len_max = max(seq_len_max, state_pred_seq.shape[0])

            loss_dict = {
                "chamfer": {"surf": [], "full": [], "synth": []},
                "emd": {"surf": [], "full": [], "synth": []},
                "h": {"surf": [], "full": [], "synth": []},
                "iou": {"full": []},
            }
            for i in range(state_seq.shape[0]):
                if not args.surface_sample:
                    state_pred = state_pred_seq[i, : args.n_particles]
                    # import pdb; pdb.set_trace()
                    target_state = state_seq[i, : args.n_particles, :3]
                    loss_dict["chamfer"]["full"].append(
                        chamfer(state_pred, target_state)
                    )
                    loss_dict["emd"]["full"].append(emd(state_pred, target_state))
                    loss_dict["h"]["full"].append(hausdorff(state_pred, target_state))
                    iou_loss = iou(
                        upsample(state_pred),
                        upsample(target_state),
                        voxel_size=0.003,
                        visualize=False,
                    )
                    loss_dict["iou"]["full"].append(iou_loss)
                    # soft_iou_loss_list.append(soft_iou(state_pred, target_state, size=8, soft=True))

                state_pred_surf = state_pred_surf_seq[i][: args.n_particles]
                target_state_surf = state_surf_seq[i, : args.n_particles, :3]
                chamfer_surf_loss = chamfer(state_pred_surf, target_state_surf)
                emd_surf_loss = emd(state_pred_surf, target_state_surf)
                h_surf_loss = hausdorff(state_pred_surf, target_state_surf)

                if args.data_type == "synthetic":
                    loss_dict["chamfer"]["synth"].append(chamfer_surf_loss)
                    loss_dict["emd"]["synth"].append(emd_surf_loss)
                    loss_dict["h"]["synth"].append(h_surf_loss)

                    state_pred = state_pred_seq[i, : args.n_particles]
                    gt_target_state = state_gt_seq[i, : args.n_particles, :3]
                    loss_dict["chamfer"]["surf"].append(
                        chamfer(state_pred, gt_target_state)
                    )
                    loss_dict["emd"]["surf"].append(emd(state_pred, gt_target_state))
                    loss_dict["h"]["surf"].append(
                        hausdorff(state_pred, gt_target_state)
                    )
                else:
                    loss_dict["chamfer"]["surf"].append(chamfer_surf_loss)
                    loss_dict["emd"]["surf"].append(emd_surf_loss)
                    loss_dict["h"]["surf"].append(h_surf_loss)

            for loss_name, loss_type_dict in loss_dict.items():
                for loss_type_name, loss_list in loss_type_dict.items():
                    if len(loss_list) > 0:
                        loss_dict_all[loss_name][loss_type_name].append(loss_list)

            if args.data_type == "synthetic":
                render_anim(
                    gnn.args,
                    [f"Eval", "Synthetic", "GT"],
                    [state_pred_seq, state_seq, state_gt_seq],
                    path=os.path.join(
                        dy_anim_path, f"{dataset}_{str(idx).zfill(3)}.mp4"
                    ),
                )
            else:
                render_anim(
                    gnn.args,
                    [f"Eval", "GT"],
                    [state_pred_seq, state_seq],
                    path=os.path.join(
                        dy_anim_path, f"{dataset}_{str(idx).zfill(3)}.mp4"
                    ),
                )

        for loss_name, loss_type_dict in loss_dict_all.items():
            for loss_type_name, loss_list_all in loss_type_dict.items():
                if len(loss_list_all) > 0:
                    for i in range(len(loss_list_all)):
                        # import pdb; pdb.set_trace()
                        pad_len = seq_len_max - len(loss_list_all[i])
                        loss_list_all[i] = np.append(
                            np.array(loss_list_all[i]),
                            np.zeros(pad_len) + loss_list_all[i][-1],
                        )

                    loss_avg = np.mean(loss_list_all, axis=0)
                    loss_std = np.std(loss_list_all, axis=0)

                    loss_name_full = f"{loss_name}_{loss_type_name}"

                    plot_eval_loss(
                        f"{loss_name_full} loss",
                        {loss_name_full: loss_avg},
                        loss_std_dict={loss_name_full: loss_std},
                        path=os.path.join(
                            dy_plot_path, f"{dataset}_{loss_name_full}_loss.png"
                        ),
                    )

                    print(
                        f"{loss_name_full}:\n\tlast frame:  {round(loss_avg[-1], 6)} (+- {round(loss_std[-1], 6)})"
                    )
                    print(
                        f"\tover frames: {round(np.mean(loss_avg), 6)} (+- {round(np.std(loss_avg), 6)})"
                    )


if __name__ == "__main__":
    args = gen_args()
    evaluate(args, load_args=True)
