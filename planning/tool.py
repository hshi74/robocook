import copy
import numpy as np
import torch

from precoded_planner import *
from learned_planner import LearnedPlanner
from model_based_planner import ModelBasedPlanner
from RL_planner import RLPlanner
from timeit import default_timer as timer
from utils.config import update_dy_args
from utils.visualize import *


class Tool(object):
    def __init__(
        self, args, name, planner_type, tool_params=None, tool_model_path_list=None
    ):
        self.name = name
        self.planner_type = planner_type

        if "precoded" in planner_type:
            tool_args = copy.deepcopy(args)
            tool_args.env = name
            if "pusher" in name:
                self.planner = PusherPlanner(tool_args)
            elif "cutter_planar" in name:
                self.planner = CutterPlanarPlanner(tool_args)
            elif "cutter_circular" in name:
                self.planner = CutterCircularPlanner(tool_args)
            elif "spatula" in name:
                self.planner = SpatulaPlanner(tool_args)
            elif "hook" in name:
                self.planner = HookPlanner(tool_args)
            else:
                raise NotImplementedError
        elif "gnn" in planner_type:
            tool_args = copy.deepcopy(args)
            tool_args_dict = np.load(
                f"{tool_model_path_list[0]}_args.npy", allow_pickle=True
            ).item()
            tool_args = update_dy_args(tool_args, name, tool_args_dict)
            tool_args.env = name
            self.planner = ModelBasedPlanner(
                tool_args, tool_params, model_path=f"{tool_model_path_list[0]}.pth"
            )
        elif "sim" in planner_type:
            tool_args = copy.deepcopy(args)
            tool_args.env = name
            self.planner = ModelBasedPlanner(tool_args, tool_params)
        elif "learned" in planner_type:
            self.planner = LearnedPlanner(
                copy.deepcopy(args), name, tool_model_path_list, tool_params
            )
        elif "RL" in planner_type:
            tool_args = copy.deepcopy(args)
            tool_args_dict = np.load(
                f"{tool_model_path_list[0]}_args.npy", allow_pickle=True
            ).item()
            tool_args = update_dy_args(tool_args, name, tool_args_dict)
            tool_args.env = name
            self.planner = RLPlanner(
                tool_args, tool_params, model_path=f"{tool_model_path_list[0]}.pth"
            )

        else:
            raise NotImplementedError

    def rollout(
        self,
        state_cur_dict,
        target_shape,
        rollout_path,
        max_n_actions,
        rs_loss_threshold=float("inf"),
    ):
        if "sim" in self.planner_type:
            state_cur = state_cur_dict["dense"]
        else:
            state_cur = state_cur_dict["tensor"]

        with torch.no_grad():
            start = timer()
            param_seq = self.planner.plan(
                state_cur,
                target_shape,
                rollout_path,
                max_n_actions,
                rs_loss_threshold=rs_loss_threshold,
            )
            end = timer()
            print(f"{self.name.upper()}: \n{param_seq}")
            print(f"PLANNING TIME: {end - start}")

            time_path = os.path.join(rollout_path, "..", "planning_time.txt")
            if os.path.exists(time_path):
                with open(time_path, "r") as f:
                    planning_time = float(f.read())
            else:
                planning_time = 0.0

            with open(time_path, "w") as f:
                f.write(str(planning_time + end - start))

            state_seq, info_dict = self.planner.eval_soln(
                param_seq, state_cur, target_shape
            )
            print(f"{self.name.upper()} LOSS: {info_dict['loss'][-1]} ")

        return param_seq, state_seq, info_dict
