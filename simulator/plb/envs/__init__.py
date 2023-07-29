import gym
from .env import PlasticineEnv
from gym import register

ENVS = []
# for env_name in ['Move', 'Torus', 'Rope', 'Writer', "Pinch",
#                  "Rollingpin", "Chopsticks", "Table", 'TripleMove', 'Assembly', 'Cutting']:
#     for id in range(5):
#         register(
#             id = f'{env_name}-v{id+1}',
#             entry_point=f"plb.envs.env:PlasticineEnv",
#             kwargs={'cfg_path': f"{env_name.lower()}.yml", "version": id+1},
#             max_episode_steps=50
#         )

env_name = 'gripper_fixed'
register(
        id = f'{env_name}-v{1}',
        entry_point=f"plb.envs.env:PlasticineEnv",
        kwargs={'cfg_path': f"{env_name}.yml", "version": 1},
        max_episode_steps=3
        )

def make(env_name, nn=False, sdf_loss=10, density_loss=10, contact_loss=1, soft_contact_loss=False):
    env: PlasticineEnv = gym.make(env_name, nn=nn)
    env.taichi_env.loss.set_weights(sdf=sdf_loss, density=density_loss,
                                    contact=contact_loss, is_soft_contact=soft_contact_loss)
    return env

def make_new(env_name, learned_model, args, use_gpu, device):
    env: PlasticineEnv = gym.make(env_name, args=args, nn=False, learned_model=learned_model, use_gpu=use_gpu, device=device)
    return env