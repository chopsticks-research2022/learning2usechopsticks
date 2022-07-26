import os

import numpy as np
import torch

from algorithm.model import load_model
from envs.hand_env_wrapper import HandEnvWrapper
from envs.hand_env import HandEnv

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def eval(hand_xml, openloop_path=None, model_path=None, render=None):
    xml =  hand_xml
    openloop_path = openloop_path
    env = HandEnvWrapper(HandEnv(xml, openloop_path, render_mode=render))
    env.reset()
    env.set_task_t(1500)
    _, actor, _ = load_model(model_path, mode='single')

    frames = []
    with torch.no_grad():
        rwd_acc = 0
        test_step=0
        current_step = 0

        obs = env.reset()
        obs= torch.Tensor(obs).float()#.view(-1,1)

        while(True):
            ac = actor(obs).numpy()
            ac = ac.reshape(-1)
            obs, rwd, done , info = env.step(ac)
            frames.append(info['frames'])
            rwd_acc += rwd

            if(done):
                print(rwd_acc)
                print(test_step)
                rwd_acc = 0 
                test_step = 0
                obs = env.reset()
                obs = torch.Tensor(obs).float()
                continue
            
            test_step+=1
            current_step +=1
            obs = torch.Tensor(obs).float()#.view(-1,1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, help="xml name")
    parser.add_argument("--traj", type=str, help="traj name")
    parser.add_argument("--model_path", type=str, help="model name")
    args = parser.parse_args()
    eval(args.xml, openloop_path=args.traj, model_path=args.model_path, render='human')
