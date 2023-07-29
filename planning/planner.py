import torch

# parametric space for each tool:

# GNN planner:
# gripper_asym: dist_to_center, rot(0-2pi), grip_width
# gripper_sym_plane: dist_to_center, rot(0-pi), grip_width
# gripper_sym_rod: dist_to_center, rot(0-pi), grip_width
# press_circle / punch_circle: press_x, press_y, press_z
# press_square / punch_square: press_x, press_y, press_z, rot(0-pi/2)
# roller_small / roller_large: roll_x, roll_y, roll_z, rot(0-pi), roll_dist

# Precoded planner:
# pusher: push_x, push_y
# cutter_planar: cut_x, cut_y, cut_rot
# cutter_circular: cut_x, cut_y
# spatula: pick_x, pick_y, place_x, place_y
# hook: none


class Planner(object):
    def plan(self, state_cur, target_shape, rollout_path):
        self.rollout_path = rollout_path
        self.center = torch.mean(state_cur.squeeze(), dim=0).cpu()
