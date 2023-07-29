import random
import numpy as np
import torch
import os
import sys

from plb.envs import make, make_new
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac

## from the other repo
from dynamics.model import Model
from utils.data_utils import count_parameters
from utils.config import gen_args

RL_ALGOS = ['sac']

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = gen_args()
    if args.rl_num_steps is None:
        args.rl_num_steps = 30000

    logger = Logger(args.rl_path)
    set_random_seed(args.rl_seed)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = Model(args, args.use_gpu)
    print("model_kp #params: %d" % count_parameters(model))
    model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter)
    args.dy_out_path = args.dy_out_path_control
    model_path = os.path.join(args.dy_out_path, model_name)
    if args.use_gpu:
        pretrained_dict = torch.load(model_path)
    else:
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()

    # only load parameters in dynamics_predictor
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if 'dynamics_predictor' in k and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    model = model.to(device)


    env = make_new(args.rl_env_name, learned_model=model, args=args, use_gpu=args.use_gpu, device=device)

    if args.rl_algo == 'sac':
        train_sac(env, args.rl_path, logger, args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
