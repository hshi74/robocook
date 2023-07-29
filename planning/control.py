import glob
import numpy as np
import open3d as o3d
import os
import pickle
import readchar
import subprocess
import sys
import torch
import yaml

torch.set_printoptions(sci_mode=False)

from planning.pcd_tool_classifier.build_dataset import process_pcd
from planning.pcd_tool_classifier.classifier import PcdClassifer
from planning.image_tool_classifier.classifier import ImageClassifer
from control_utils import *
from datetime import datetime
from std_msgs.msg import UInt8
from tool import *
from utils.config import gen_args
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


args = gen_args()

if args.close_loop:
    import rospy
    from perception.sample_pcd import ros_bag_to_pcd
    from rospy.numpy_msg import numpy_msg
    from rospy_tutorials.msg import Floats
    from std_msgs.msg import UInt8, String


command_feedback = 0


def command_fb_callback(msg):
    global command_feedback
    if msg.data > 0:
        command_feedback = msg.data


def get_test_name(args):
    test_name = ["control"]
    if args.close_loop:
        test_name.append("close")
    else:
        test_name.append("open")

    if len(args.active_tool_list) == 1:
        if "rm=1" in args.tool_model_dict[args.tool_type]:
            test_name.append("rm")
        if "attn=1" in args.tool_model_dict[args.tool_type]:
            test_name.append("attn")

    test_name.append(f"max={args.max_n_actions}")
    test_name.append(args.optim_algo)
    if "CEM" in args.optim_algo and not args.debug:
        test_name.append(f"{args.CEM_sample_size}")
        test_name.append(f"{args.CEM_decay_factor}")
    test_name.append(args.control_loss_type)

    if args.debug:
        test_name.append("debug")

    test_name.append(datetime.now().strftime("%b-%d-%H:%M:%S"))

    return "_".join(test_name)


cd = os.path.dirname(os.path.realpath(sys.argv[0]))

rollout_root = os.path.join(
    cd,
    "..",
    "dump",
    "control",
    f"control_{args.tool_type}",
    args.target_shape_name,
    get_test_name(args),
)
os.system("mkdir -p " + rollout_root)

for dir in ["states", "raw_data"]:
    os.system("mkdir -p " + os.path.join(rollout_root, dir))


class MPController(object):
    def __init__(self):
        self.get_target_shapes()
        self.load_tools()
        if "image" in args.cls_type:
            self.classifier = ImageClassifer(args)
        else:
            self.classifier = PcdClassifer(args)

    def get_target_shapes(self):
        target_dir = os.path.join(cd, "..", "target_shapes", args.target_shape_name)
        if "sim" in args.target_shape_name:
            target_list = [os.path.basename(args.target_shape_name)]
        else:
            target_list = sorted(
                [d.name for d in os.scandir(target_dir) if d.is_dir()]
            )[1:]

        self.target_shapes = []
        for target in target_list:
            if "sim" in args.target_shape_name:
                prefix = target
            else:
                prefix = f"{target}/{target.split('_')[0]}"

            target_shape = {}
            target_shape["label"] = "_".join(target.split("_")[1:])
            for type in ["sparse", "dense", "surf"]:
                if type == "sparse":
                    target_frame_path = os.path.join(target_dir, f"{prefix}.h5")
                else:
                    target_frame_path = os.path.join(target_dir, f"{prefix}_{type}.h5")

                if os.path.exists(target_frame_path):
                    target_data = load_data(args.data_names, target_frame_path)
                    target_shape[type] = target_data[0]

            raw_pcd_path = os.path.join(target_dir, f"{prefix}_raw.ply")
            if os.path.exists(raw_pcd_path):
                target_shape["raw_pcd"] = o3d.io.read_point_cloud(raw_pcd_path)

            if not "sim" in args.target_shape_name:
                image_paths = [
                    os.path.join(
                        target_dir, target, f"{target.split('_')[0]}_cam_{x+1}.png"
                    )
                    for x in range(4)
                ]
                target_shape["images"] = image_paths

            self.target_shapes.append(target_shape)

    def load_tools(self):
        self.tool_name_list = args.tool_name_list
        if args.active_tool_list == "default":
            self.active_tool_name_list = self.tool_name_list
        else:
            self.active_tool_name_list = args.active_tool_list.split("+")

        if "dumpling" in args.target_shape_name:
            self.active_tool_name_list = [
                "cutter_circular",
                "cutter_planar",
                "gripper_sym_plane",
                "hook",
                "press_square",
                "pusher",
                "roller_large",
                "spatula_large",
                "spatula_small",
            ]
        elif "alphabet" in args.target_shape_name:
            self.active_tool_name_list = [
                "gripper_asym",
                "gripper_sym_plane",
                "gripper_sym_rod",
                "punch_circle",
                "punch_square",
            ]
        else:
            raise NotImplementedError

        # print(f"[INFO] The active tool list is: {self.active_tool_name_list}")

        with open("config/tool_plan_params.yml", "r") as f:
            tool_params = yaml.load(f, Loader=yaml.FullLoader)

        with open("config/tool_model_map.yml", "r") as f:
            tool_model_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.active_tool_dict = {}
        for tool_name in self.active_tool_name_list:
            if tool_name in args.precoded_tool_list:
                tool = Tool(args, tool_name, "precoded")
            else:
                if "sim" in args.planner_type:
                    tool_model_path_list = None
                else:
                    if args.planner_type == "learned":
                        type = "learned"
                    else:
                        type = "gnn"
                    tool_model_names = tool_model_dict[type][tool_name]
                    if isinstance(tool_model_names, list):
                        tool_model_path_list = []
                        for tool_model_name in tool_model_names:
                            tool_model_path_list.append(
                                os.path.join(
                                    cd,
                                    "..",
                                    "models",
                                    type,
                                    tool_model_name,
                                )
                            )
                    else:
                        tool_model_path_list = [
                            os.path.join(cd, "..", "models", type, tool_model_names)
                        ]
                tool = Tool(
                    args,
                    tool_name,
                    args.planner_type,
                    tool_params[tool_name],
                    tool_model_path_list,
                )

            self.active_tool_dict[tool_name] = tool

    def get_state_from_ros(self, tool_name, ros_data_path, rgb=True):
        command_time = datetime.now().strftime("%b-%d-%H:%M:%S")
        ros_pcd_path_prefix = os.path.join(ros_data_path, command_time)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if rgb:
                self.command_pub.publish((String(f"{command_time}.shoot.pcd+rgb")))
            else:
                self.command_pub.publish((String(f"{command_time}.shoot.pcd")))

            self.ros_data_path_pub.publish(String(ros_pcd_path_prefix))
            if os.path.exists(ros_pcd_path_prefix + ".bag"):
                print(f"[INFO] Received data from cameras!")
                break

            rate.sleep()

        vis_args = copy.deepcopy(args)
        vis_args.env = tool_name
        state_cur_pcd = process_pcd(
            vis_args, ros_pcd_path_prefix + ".bag", visualize=False
        )
        pcd_dense, pcd_sparse, state_cur = ros_bag_to_pcd(
            vis_args, ros_pcd_path_prefix + ".bag", visualize=False
        )

        state_cur_dict = {
            "raw_pcd": state_cur_pcd,
            "tensor": torch.tensor(
                state_cur[: args.n_particles], device=args.device, dtype=torch.float32
            ).unsqueeze(0),
            "dense": torch.tensor(
                np.asarray(pcd_dense.points), device=args.device, dtype=torch.float32
            ).unsqueeze(0),
        }

        if rgb:
            ros_image_paths = [f"{ros_pcd_path_prefix}_cam_{x+1}.png" for x in range(4)]
            state_cur_dict["images"] = ros_image_paths

        return state_cur_dict

    def control(self):
        if args.close_loop:
            ros_data_path = os.path.join(rollout_root, "raw_data")
            # args.env = 'hook'
            state_init_dict = self.get_state_from_ros(args.env, ros_data_path)
        else:
            target_dir = os.path.join(cd, "..", "target_shapes", args.target_shape_name)

            if "sim" in args.target_shape_name:
                name = "start"
            else:
                name = "000"

            init_state_path = os.path.join(target_dir, name, f"{name}.h5")
            dense_init_state_path = os.path.join(target_dir, name, f"{name}_dense.h5")
            surf_init_state_path = os.path.join(target_dir, name, f"{name}_surf.h5")
            raw_pcd_path = os.path.join(target_dir, name, f"{name}_raw.ply")

            init_state_data = load_data(args.data_names, init_state_path)
            dense_init_state_data = load_data(args.data_names, dense_init_state_path)
            surf_init_state_data = load_data(args.data_names, surf_init_state_path)
            raw_pcd = o3d.io.read_point_cloud(raw_pcd_path)

            image_paths = [
                os.path.join(target_dir, name, f"{name}_cam_{x+1}.png")
                for x in range(4)
            ]

            init_state = {
                "sparse": init_state_data[0],
                "dense": dense_init_state_data[0],
                "surf": surf_init_state_data[0],
                "images": image_paths,
                "raw_pcd": raw_pcd,
            }

            state_init_dict = {
                "tensor": torch.tensor(
                    init_state["surf"][: args.n_particles],
                    device=args.device,
                    dtype=torch.float32,
                ).unsqueeze(0),
                "images": init_state["images"],
                "raw_pcd": init_state["raw_pcd"],
                "dense": torch.tensor(
                    init_state["dense"][: -args.floor_dim],
                    device=args.device,
                    dtype=torch.float32,
                ).unsqueeze(0),
            }

        tool_list = []
        param_seq_dict = {}
        state_seq_list = []
        info_dict_list = []
        loss_dict = {"Chamfer": [], "EMD": [], "IOU": []}
        state_cur_dict = state_init_dict
        step = 0
        while True:
            # if no subtarget, always use the last target as target
            # chamfer_loss, emd_loss, iou_loss = self.eval_state(state_cur_dict['tensor'].cpu(), step, plan_target_idx)

            # loss_dict['Chamfer'].append(round(chamfer_loss, 4))
            # loss_dict['EMD'].append(round(emd_loss, 4))
            # loss_dict['IOU'].append(round(iou_loss, 4))

            # if (args.subtarget or not args.close_loop) and
            # if step == len(self.target_shapes): break

            print(f"{bcolors.HEADER}{'#'*30} STEP {step} {'#'*30}{bcolors.ENDC}")
            step_folder_name = f"{str(step).zfill(3)}"
            rollout_path = os.path.join(rollout_root, step_folder_name)
            for dir in [
                "param_seqs",
                "anim",
                "anim_args",
                "states",
                "optim_plots",
                "raw_data",
                "cls_plots",
            ]:
                os.system("mkdir -p " + os.path.join(rollout_path, dir))

            # classify which tool to use
            cls_target_idx = step if args.subtarget else len(self.target_shapes) - 1
            tools_pred = self.classify(state_cur_dict, cls_target_idx, rollout_path)
            # tools_pred = ['roller_large']

            # plan the actions given the tool
            tool_name, param_seq, state_seq, info_dict, state_cur_dict = self.plan(
                tools_pred, state_cur_dict, step, rollout_path
            )

            tool_list.append(tool_name)
            param_seq_dict[f"{step}-{tool_name}"] = param_seq.tolist()
            state_seq_list.append(state_seq)
            info_dict_list.append(info_dict)

            print(f"{'#'*24} SUMMARY OF STEP {step} {'#'*25}")
            print(f"USE {tool_name.upper()}: \n{param_seq}")
            print(f"LOSS AFTER STEP {step}: {info_dict['loss'][-1]}")

            # use 0.01 as the threshold
            if not args.close_loop or info_dict["loss"][-1] < 0.015:
                step += 1

            if (
                "dumpling" in args.target_shape_name and info_dict["loss"][-1] == 0.0
            ) or (
                not "dumpling" in args.target_shape_name
                and step == len(self.target_shapes)
            ):
                print("ALL DONE!!!")
                break

            # if args.close_loop:
            #     ros_data_path = os.path.join(rollout_root, 'raw_data')
            #     state_cur_dict = self.get_state_from_ros(args, ros_data_path)
            # else:
            #     state_cur_dict['tensor'] = torch.tensor(state_seq[-1][:args.n_particles],
            #         device=args.device, dtype=torch.float32).unsqueeze(0)
            #     state_cur_dict['images'] = self.target_shapes[step]['images']

        data = [tool_list, loss_dict, param_seq_dict, state_seq_list, info_dict_list]
        self.summary(data)

    def classify(self, state_cur_dict, target_idx, rollout_path):
        tools_pred = self.classifier.eval(
            state_cur_dict,
            self.target_shapes[target_idx],
            path=os.path.join(rollout_path, "cls_plots"),
        )

        print(f"[OUT] The classifier predicts that we should use: {tools_pred}")

        # augment the predictions with tools from the same family
        # tools_pred_aug = []
        # for tool_name in tools_pred:
        #     if 'gripper' in tool_name:
        #         tools_pred_aug.extend(['gripper_asym', 'gripper_sym_plane', 'gripper_sym_rod'])
        #     elif 'press' in tool_name or 'punch' in tool_name:
        #         if 'large' in tool_name:
        #             tools_pred_aug.extend(['press_square', 'press_circle'])
        #         else:
        #             tools_pred_aug.extend(['punch_square', 'punch_circle'])
        #     elif 'roller' in tool_name:
        #         tools_pred_aug.extend(['roller_large', 'roller_small'])
        #     else:
        #         tools_pred_aug.append(tool_name)

        # if len(tools_pred_aug) > len(tools_pred):
        #     print(f'[OUT] But we should try: {tools_pred_aug}')

        return tools_pred

    def execute(self, tool_name, param_seq):
        print(f"Executing {tool_name.upper()}...")
        # tool = self.active_tool_dict[tool_name]
        # if 'gripper' in tool_name:
        #     param_seq_temp = []
        #     for i in range(param_seq.shape[0]):
        #         param_seq_temp.append(np.concatenate((tool.planner.center.numpy()[:2], param_seq[i])))
        #     param_seq = np.array(param_seq_temp)
        self.param_seq_pub.publish(param_seq.flatten().astype(np.float32))
        command_time = datetime.now().strftime("%b-%d-%H:%M:%S")
        self.command_pub.publish(String(f"{command_time}.run.{tool_name}"))

    def plan(self, tools_pred, state_cur_dict, step, rollout_path):
        global command_feedback

        target_idx_list = []
        for tool_name in tools_pred:
            if args.subtarget:
                target_idx = step
            else:
                target_idx = -1
                for i in range(len(self.target_shapes)):
                    if self.target_shapes[i]["label"] == tool_name:
                        target_idx = i
                        break
            target_idx_list.append(target_idx)

        target_idx_max = max(target_idx_list)
        for i in range(len(target_idx_list)):
            if target_idx_list[i] == -1:
                target_idx_list[i] = target_idx_max

        best_target_idx = 0
        best_loss = float("inf")
        for tool_name, target_idx in zip(tools_pred, target_idx_list):
            # if not 'roller' in tool_name:
            #     continue

            if not tool_name in self.active_tool_name_list:
                print(
                    f"{bcolors.WARNING}[WARNING] {tool_name} is inactive!{bcolors.ENDC}"
                )
                continue

            print(f"{'#'*15} {tool_name.upper()} {'#'*15}")
            if tool_name in args.precoded_tool_list:
                rs_loss_threshold = float("inf")
            else:
                rs_loss_threshold = best_loss + 0.001

            param_seq, state_seq, info_dict = self.active_tool_dict[tool_name].rollout(
                state_cur_dict,
                self.target_shapes[target_idx],
                rollout_path,
                args.max_n_actions,
                rs_loss_threshold=rs_loss_threshold,
            )

            loss_weighted = info_dict["loss"][-1]  # / (target_idx + 1)
            # print(f"The weighted loss of {tool_name} is {loss_weighted}!")
            if loss_weighted < best_loss:
                best_tool_name = tool_name
                best_target_idx = target_idx
                best_param_seq = param_seq
                best_state_seq = state_seq
                best_info_dict = info_dict
                best_loss = loss_weighted

        if args.close_loop:
            # print(param_seq.shape, state_seq.shape)
            param_seq = best_param_seq
            state_seq = best_state_seq
            info_dict = best_info_dict

            act_len = state_seq.shape[0] // param_seq.shape[0]
            act_start = 0
            act_end = 1
            param_seq_todo = param_seq[act_start:act_end].numpy()
            param_seq_pred = param_seq_todo
            state_seq_pred = state_seq[:act_len][: args.n_particles]
            state_pred_tensor = torch.tensor(
                state_seq_pred[-1], device=args.device, dtype=torch.float32
            ).unsqueeze(0)
            info_dict_pred = info_dict

            if "roller" in best_tool_name:
                max_n_actions = 5
                pred_err_bar = 0.05
            else:
                max_n_actions = 4
                pred_err_bar = 0.02

            ros_data_path = os.path.join(rollout_path, "raw_data")
            # loss_dict = {'Chamfer': [], 'EMD': [], 'IOU': []}
            while not rospy.is_shutdown():
                self.execute(best_tool_name, param_seq_todo)

                while command_feedback != 1:
                    continue

                command_feedback = 0

                print("Waiting for disturbance... Press enter when you finish...")
                readchar.readkey()

                state_cur_dict = self.get_state_from_ros(best_tool_name, ros_data_path)

                pred_err = chamfer(
                    state_cur_dict["tensor"].squeeze(),
                    state_pred_tensor.squeeze(),
                    pkg="torch",
                )
                print(f"The prediction error is {pred_err}!")
                # chamfer_loss, emd_loss = self.eval_state(state_cur_dict['tensor'].cpu(),
                #     step, best_target_idx, state_pred=state_pred_tensor.cpu(), pred_err=pred_err)

                if (
                    not best_tool_name in args.precoded_tool_list
                    and param_seq_pred.shape[0] < max_n_actions
                ):
                    # TODO: Need to tune this number
                    if pred_err > 0 and pred_err < pred_err_bar:
                        print(f"The prediction is good enough!")
                        if act_end < param_seq.shape[0]:
                            # move to the next action
                            act_start = act_end
                        elif "roller" in best_tool_name:
                            param_seq, state_seq, info_dict = self.active_tool_dict[
                                best_tool_name
                            ].rollout(
                                state_cur_dict,
                                self.target_shapes[best_target_idx],
                                rollout_path,
                                min(
                                    max_n_actions - param_seq_pred.shape[0],
                                    args.max_n_actions,
                                ),
                            )
                            act_start = 0
                        else:
                            break
                    else:
                        # figure out a new solution
                        param_seq, state_seq, info_dict = self.active_tool_dict[
                            best_tool_name
                        ].rollout(
                            state_cur_dict,
                            self.target_shapes[best_target_idx],
                            rollout_path,
                            min(
                                max_n_actions - param_seq_pred.shape[0],
                                args.max_n_actions,
                            ),
                        )
                        act_start = 0

                    act_end = act_start + 1

                    param_seq_todo = param_seq[act_start:act_end].numpy()
                    param_seq_pred = np.concatenate((param_seq_pred, param_seq_todo))
                    state_seq_pred = np.concatenate(
                        (
                            state_seq_pred,
                            state_seq[act_start * act_len : act_end * act_len],
                        )
                    )
                    state_pred_tensor = torch.tensor(
                        state_seq_pred[-1], device=args.device, dtype=torch.float32
                    ).unsqueeze(0)

                    for key, value in info_dict.items():
                        info_dict_pred[key].extend(value)
                else:
                    break

            best_param_seq = param_seq_pred
            best_state_seq = state_seq_pred
            best_info_dict = info_dict_pred
        else:
            state_cur_dict["tensor"] = torch.tensor(
                best_state_seq[-1][: args.n_particles],
                device=args.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            # state_cur_dict['tensor'] = torch.tensor(self.target_shapes[step]['surf'],
            #     device=args.device, dtype=torch.float32).unsqueeze(0)
            state_cur_dict["images"] = self.target_shapes[step]["images"]
            state_cur_dict["raw_pcd"] = self.target_shapes[step]["raw_pcd"]

        return (
            best_tool_name,
            best_param_seq,
            best_state_seq,
            best_info_dict,
            state_cur_dict,
        )

    def summary(self, data):
        tool_list, loss_dict, param_seq_dict, state_seq_list, info_dict_list = data

        print(f"{'#'*27} MPC SUMMARY {'#'*28}")
        for key, value in param_seq_dict.items():
            print(f"{key}: {value}")

        if args.close_loop:
            state_cur_dict = self.get_state_from_ros(
                tool_list[-1], os.path.join(rollout_root, "raw_data")
            )
            state_cur = state_cur_dict["tensor"].squeeze().cpu().numpy()
        else:
            state_cur = state_seq_list[-1][-1, : args.n_particles]

        state_cur_norm = state_cur - np.mean(state_cur, axis=0)
        state_goal = self.target_shapes[-1]["surf"]
        state_goal_norm = state_goal - np.mean(state_goal, axis=0)
        dist_final = chamfer(state_cur_norm, state_goal_norm, pkg="numpy")
        print(f"FINAL chamfer distance: {dist_final}")

        with open(os.path.join(rollout_root, "planning_time.txt"), "r") as f:
            print(f"TOTAL planning time (s): {f.read()}")

        with open(os.path.join(rollout_root, f"MPC_param_seq.yml"), "w") as f:
            yaml.dump(param_seq_dict, f, default_flow_style=True)

        for info_dict in info_dict_list:
            for p in info_dict["subprocess"]:
                p.communicate()

        anim_list_path = os.path.join(rollout_root, f"anim_list.txt")
        with open(anim_list_path, "w") as f:
            for i, tool in enumerate(tool_list):
                anim_path_list = sorted(
                    glob.glob(
                        os.path.join(rollout_root, str(i).zfill(3), "anim", "*.mp4")
                    )
                )
                for anim_path in anim_path_list:
                    anim_name = os.path.basename(anim_path)
                    if (
                        tool in anim_name
                        and not "RS" in anim_name
                        and not "CEM" in anim_name
                        and not "GD" in anim_name
                        and not "sim" in anim_name
                    ):
                        f.write(f"file '{anim_path}'\n")

        mpc_anim_path = os.path.join(rollout_root, f"MPC_anim.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                anim_list_path,
                "-c",
                "copy",
                mpc_anim_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def main():
    tee = Tee(os.path.join(rollout_root, "control.txt"), "w")

    mpcontroller = MPController()

    if args.close_loop:
        rospy.init_node("control", anonymous=True)

        mpcontroller.param_seq_pub = rospy.Publisher(
            "/param_seq", numpy_msg(Floats), queue_size=10
        )
        mpcontroller.command_pub = rospy.Publisher("/command", String, queue_size=10)
        mpcontroller.ros_data_path_pub = rospy.Publisher(
            "/raw_data_path", String, queue_size=10
        )
        rospy.Subscriber("/command_feedback", UInt8, command_fb_callback)

    mpcontroller.control()


if __name__ == "__main__":
    main()
