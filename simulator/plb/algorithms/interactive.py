########## This is the file to let human control the gripper ############

import pygame
import numpy as np
import torch

# shared across tasks
from plb.optimizer.optim import Adam
from plb.engine.taichi_env import TaichiEnv
from plb.config.default_config import get_cfg_defaults, CN

from plb.algorithms.sample_data import em_distance, chamfer_distance, load_data

import os
import cv2
import taichi as ti
ti.init(arch=ti.gpu)
from plb.config import load
import sys

from datetime import datetime

cwd = os.getcwd()
root_dir = cwd

robot_emd_loss = {'C': 0.0383, 'E': 0.0402, 'Y': 0.0412, 'Z': 0.0360, 'heart': 0.0238}
robot_chamfer_loss = {'C': 0.0440, 'E': 0.0464, 'Y': 0.0419, 'Z': 0.0390, 'heart': 0.0348}

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


def rotate_around_y(prim1_pos, prim2_pos, rot_unit):
    # rotation function
    mid_point = (prim1_pos[:3] + prim2_pos[:3]) / 2
    angle_cur = np.arctan2(prim1_pos[2] - prim2_pos[2], prim1_pos[0] - prim2_pos[0])
    angle_cur = np.pi - angle_cur
    r = np.linalg.norm(prim1_pos - prim2_pos) / 2

    x1 = mid_point[0] - r * np.cos(angle_cur + rot_unit) - prim1_pos[0]
    z1 = mid_point[2] + r * np.sin(angle_cur + rot_unit) - prim1_pos[2]
    x2 = mid_point[0] + r * np.cos(angle_cur + rot_unit) - prim2_pos[0]
    z2 = mid_point[2] - r * np.sin(angle_cur + rot_unit) - prim2_pos[2]

    delta_a = np.concatenate([np.array([x1, 0, z1]), np.zeros(3), 
                              np.array([x2, 0, z2]), np.zeros(3)])
    return delta_a


def set_parameters(env: TaichiEnv, yield_stress, E, nu):
    # set parameters for the environment
    env.simulator.yield_stress.fill(yield_stress)
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    env.simulator.mu.fill(_mu)
    env.simulator.lam.fill(_lam)


def update_camera(env):
    # camera extrinsics
    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)
    env.render_cfg.defrost()
    env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.2)
    env.render_cfg.camera_rot_1 = (0.8, 0.)
    env.render_cfg.camera_pos_2 = (2.4, 2.5, 0.2)
    env.render_cfg.camera_rot_2 = (0.8, 1.8)
    env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0.2)
    env.render_cfg.camera_rot_3 = (0.8, -1.8)
    env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
    env.render_cfg.camera_rot_4 = (0.8, 3.14)    


data_names = ['positions', 'shape_quats', 'scene_params']
def save_env(env, rollout_dir, shape_name, frame, actions, n_particles=300):
    x = env.simulator.get_x(0)
    step_size = len(x) // n_particles
    x = x[::step_size]
    x = x[:n_particles]

    target = load_data(data_names, f"{root_dir}/interactive/target/{shape_name}.h5")[0][:n_particles]

    emd_loss = em_distance(torch.tensor(x), torch.tensor(target))
    chamfer_loss = chamfer_distance(torch.tensor(x), torch.tensor(target))
    print(f"Test results:\nEMD: {emd_loss}\nChamfer: {chamfer_loss}")
    with open(f"{rollout_dir}/{shape_name}_loss.npy", 'wb') as f:
        np.save(f, np.array([emd_loss, chamfer_loss]))

    primitive_state = [env.primitives.primitives[0].get_state(0), env.primitives.primitives[1].get_state(0)]
    rgb, depth = env.render('img')

    img = env.render_multi(mode='rgb_array', spp=3)
    rgb, depth = img[0], img[1]
    
    for num_cam in range(4):
        cv2.imwrite(f"{rollout_dir}/{shape_name}_rgb_{num_cam}.png", rgb[num_cam][..., ::-1])
    with open(f"{rollout_dir}/{shape_name}_depth_prim.npy", 'wb') as f:
        np.save(f, depth + primitive_state)
    with open(f"{rollout_dir}/{shape_name}_gtp.npy", 'wb') as f:
        np.save(f, x)
    with open(f"{rollout_dir}/{shape_name}_actions.npy", 'wb') as f:
        np.save(f, actions)


def main():
    cfg = load("../envs/gripper_fixed.yml")
    print(cfg)
    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()
    env.set_state(**state)
    print(env.renderer.camera_pos)

    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)


    unit_quat = np.array([1, 0, 0, 0])
    prim1_pos = np.array([0.3, 0.14, 0.5])
    prim2_pos = np.array([0.7, 0.14, 0.5])
    env.primitives.primitives[0].set_state(0, np.concatenate((prim1_pos, unit_quat)))
    env.primitives.primitives[1].set_state(0, np.concatenate((prim2_pos, unit_quat)))

    set_parameters(env, yield_stress=200, E=5e3, nu=0.2)
    update_camera(env)

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    pygame.init()
    size = (512, 512)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Taichi Env")

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # set up the initial scene
    screen.fill(BLACK)
    rgb_img, _ = env.render('img')
    surf = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.update()


    name = input('Please enter the name of the tester below:\n')
    shape_name = input('Please enter the name of the target shape below:\n')

    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    rollout_dir = f"{root_dir}/interactive/{name}/{shape_name}-{time_now}"
    os.system('mkdir -p ' + f"{rollout_dir}")

    print("Please click the pygame window to give it focus!!")

    tee = Tee(os.path.join(rollout_dir, 'control.txt'), 'w')

    if shape_name in 'EZ':
        env.primitives.primitives[0].r[None] = 0.03
        env.primitives.primitives[1].r[None] = 0.03

    frame = 0

    pos_unit = 0.01
    rot_unit = np.pi / 48
    actions = []
    command = ''
    # -------- Main Program Loop -----------
    while True:
        # --- Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Close the window and quit.
                pygame.quit()

            # --- Game logic should go here
            if event.type == pygame.KEYDOWN:
                delta_a = np.zeros(12)
                if event.key == pygame.K_e:
                    command = 'end'
                    save_env(env, rollout_dir, shape_name, frame, actions)
                    frame += 1
                
                if event.key == pygame.K_r:
                    command = 'reset'
                    env.set_state(**state)
                    env.primitives.primitives[0].set_state(0, np.concatenate((prim1_pos, unit_quat)))
                    env.primitives.primitives[1].set_state(0, np.concatenate((prim2_pos, unit_quat)))

                if event.key == pygame.K_d:
                    command = 'move: pos +x'
                    unit_a = np.array([pos_unit, 0, 0])
                    delta_a = np.concatenate([unit_a, np.zeros(3), unit_a, np.zeros(3)])
                if event.key == pygame.K_a:
                    command = 'move: pos -x'
                    unit_a = np.array([-pos_unit, 0, 0])
                    delta_a = np.concatenate([unit_a, np.zeros(3), unit_a, np.zeros(3)])

                if event.key == pygame.K_s:
                    command = 'move: pos +z'
                    unit_a = np.array([0, 0, pos_unit])
                    delta_a = np.concatenate([unit_a, np.zeros(3), unit_a, np.zeros(3)])
                if event.key == pygame.K_w:
                    command = 'move: pos -z'
                    unit_a = np.array([0, 0, -pos_unit])
                    delta_a = np.concatenate([unit_a, np.zeros(3), unit_a, np.zeros(3)])

                if event.key == pygame.K_1:
                    command = 'move: rot +y'
                    delta_a = rotate_around_y(prim1_pos, prim2_pos, rot_unit)
                if event.key == pygame.K_2:
                    command = 'move: rot -y'
                    delta_a = rotate_around_y(prim1_pos, prim2_pos, -rot_unit)
        
                if event.key == pygame.K_COMMA:
                    command = 'move: close'
                    angle_cur = np.arctan2(prim1_pos[2] - prim2_pos[2], prim1_pos[0] - prim2_pos[0])
                    angle_cur = np.pi - angle_cur
                    unit_a = np.array([pos_unit * np.cos(angle_cur), 0, -pos_unit * np.sin(angle_cur)])
                    delta_a = np.concatenate([unit_a, np.zeros(3), -unit_a, np.zeros(3)])
                if event.key == pygame.K_PERIOD:
                    command = 'move: open'
                    angle_cur = np.arctan2(prim1_pos[2] - prim2_pos[2], prim1_pos[0] - prim2_pos[0])
                    angle_cur = np.pi - angle_cur
                    unit_a = np.array([-pos_unit * np.cos(angle_cur), 0, pos_unit * np.sin(angle_cur)])
                    delta_a = np.concatenate([unit_a, np.zeros(3), -unit_a, np.zeros(3)])

                print(f'command: {command}')
                if command[:4] == 'move':
                    actions.append(delta_a)
                    env.step(delta_a / 0.02)

                    prim1_pos += delta_a[:3]
                    prim2_pos += delta_a[6:9]
                    print(f'prim1_pose: {prim1_pos}')

                # --- Screen-clearing code goes here
                screen.fill(BLACK)

                # --- Drawing code should go here
                # env.render('human')
                rgb_img, _ = env.render('img')
                surf = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
                screen.blit(surf, (0, 0))

                # --- Go ahead and update the screen with what we've drawn.
                pygame.display.update()

if __name__ == '__main__':
    main()