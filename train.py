import os

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from envs.hand_env_wrapper import HandEnvWrapper
from algorithm.PPO_vec import PPO_vec
from envs.hand_env import HandEnv

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train(openloop = None, xml=None, num_threads = 4, num_iter = 20000, model_load_path = None, actor_type = 'single', experts_path = None, fix_experts = False, log_dir = None):
    #take the position parameter as inputs and output the adapted kinematic pose and returned reward
    num_envs = num_threads
    v_max =  15/(1-0.99)
    v_min = 0
    time_limit = 100
    hand_xml = xml
    openloop_path = openloop
    gating_path = './data/models/gating/gating_init.tar'

    env_id = HandEnv
    env_kwargs = {
        'model_path': hand_xml,
        'openloop_path': openloop_path,
        'mode': 'batch'
    }

    vec_env = make_vec_env(
        env_id=env_id, 
        n_envs=num_envs,
        wrapper_class=HandEnvWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )
    vec_env.env_method('set_task_t', time_limit)
    vec_env.reset()
    importance_sampling = True
    PPO_vec(
           vec_env = vec_env,
           exp_id =  0,
           save_dir = log_dir + '/',
           checkpoint_batch=100,
           test_batch=100,
           gamma= 0.99,
           lam=0.95,
           v_max = v_max,
           v_min = v_min,
           time_limit= time_limit,
           CL = False,
           noise = 0.1,
           noise_mask = np.array([0.1]*7 + [1]*(vec_env.action_space.shape[0]-7)),
           importance_sampling=importance_sampling,
           num_iter = num_iter,
           model_load_path = model_load_path,
           actor_type= actor_type,
           experts_path= experts_path,
           gating_path = gating_path,
           fix_experts=fix_experts,
           log_dir = log_dir     
       )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, help="log path")
    parser.add_argument("--xml", type=str, help="xml name")
    parser.add_argument("--traj", type=str, help="traj name")
    parser.add_argument("--threads", type=int, help="the number of threads", default=4)
    args = parser.parse_args()
    train(args.traj, args.xml, args.threads, actor_type='single', log_dir=args.logdir)
