import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from .runner import GAERunner
from .data_tools import PPO_Dataset
import numpy as np
import time
from .model import *
import os
import random
from IPython import embed


def calc_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad == True:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1./2)
    return total_norm


class MotionBoundSchedule(object):
    def __init__(self, delta_reward, delta_bound):
        self.delta_reward = delta_reward
        self.delta_bound = delta_bound
        self.acc_rwd = 0

    def update(self, avg_rwd):
        self.acc_rwd += avg_rwd
        if(self.acc_rwd > self.delta_reward):
            self.acc_rwd = 0
            return True
        else:
            return False

class EpisodeScheduler(object):
    def __init__(self, init_interval = 2000, appeal_interval = 500):
        self.init_interval = init_interval
        self.appeal_interval = appeal_interval

    def update(self, it):
        if(it >= self.init_interval):
            if(it % self.appeal_interval == 0):
                return True
            else:
                return False
        else:
            return False



def importance_sampler(runner):
    #compute the test reward for each task
    num_task = runner.env.env_method('get_num_task', indices=0)[0]

    rwd_task = []
    rwd_perstep_task = []
    for i in range(num_task):
        print("idx:{}".format(i))
        runner.env.env_method('set_task_idx', i)
        _, test_rwd, rwd_perstep, _ , _ = runner.test()
        episode_length = runner.env.env_method('get_episode_length', indices=0)[0]
        print('episode length:{}'.format(episode_length))
        rwd_task.append(test_rwd)
        rwd_perstep_task.append(test_rwd / episode_length)
    runner.env.env_method('set_sample_mode', 'random')
    #from IPython import embed
    #embed()
    return np.array(rwd_perstep_task)


    
def set_seed(seed):
    '''
    set random seeds for deep reinforcement learning
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def PPO_vec(vec_env, exp_id=0,
        save_dir="./experiments",
        sample_size=20000,
        epoch_size=10,
        batch_size=256,
        checkpoint_batch=-1,
        test_batch=20,
        gamma=0.99,
        lam=0.95,
        clip_threshold=0.2,
        actor_lr=3e-5,
        critic_lr=3e-4,
        actor_wdecay = 5.0e-4,
        critic_wdecay = 5.0e-4,
        actor_momentum = 0.9,
        critic_momentum = 0.9,
        max_grad_norm= 0.5,
        v_max = 1,
        v_min = 0,
        time_limit = 500,
        use_gpu_model=False,
        CL = False,
        noise = 0.1, 
        noise_mask = None,
        importance_sampling = False, 
        seed = 0,
        num_iter = 1000,
        model_load_path = None,
        actor_type = 'single',
        experts_path = None,
        gating_path = None, 
        fix_experts = False,
        log_dir = None):

        #log parameters
        if(not os.path.exists(save_dir)):
            os.makedirs(save_dir) 
        model_path = save_dir
        total_sample = 0
        train_sample = 0
        model_path_final = None



        #network settings
        if(actor_type == 'single'):
            if(model_load_path == None):
                s_norm = Normalizer(vec_env.observation_space.shape[0])
                actor = Actor(vec_env.observation_space.shape[0], vec_env.action_space.shape[0], vec_env.action_space, hidden=[64, 64], noise=noise, noise_mask=noise_mask)
                critic = Critic(vec_env.observation_space.shape[0], v_min, v_max, hidden =[64, 64])
            else:
                s_norm, actor, critic= load_model(model_load_path)
                actor.set_std(noise * noise_mask)
        elif(actor_type == 'MOE'):
            if(model_load_path == None):
                s_norm = Normalizer(vec_env.observation_space.shape[0])
                #actor = Actor(vec_env.observation_space.shape[0], vec_env.action_space.shape[0], vec_env.action_space, hidden=[64, 64], noise=noise, noise_mask=noise_mask)
                actor = MOE(vec_env.observation_space.shape[0], 4, vec_env.action_space.shape[0], vec_env.action_space, [64, 64], [64, 64],  noise=noise, noise_mask=noise_mask)
                if(experts_path!= None):
                    actor.load_experts(experts_path)
                if(gating_path!= None):
                    actor.load_gating(gating_path)
                if(fix_experts):
                    actor.fix_experts()
                critic = Critic(vec_env.observation_space.shape[0], v_min, v_max, hidden =[64, 64])
            else:
                s_norm, actor, critic= load_model(model_load_path, 'MOE')
                #actor.set_std(noise * noise_mask)
        else:
            raise NotImplementedError
        runner = GAERunner(vec_env, s_norm, actor, critic, sample_size, gamma, lam, v_max, v_min,
            use_gpu_model = use_gpu_model)

        # optimizer
        #self.actor_optim = optim.SGD(filter(lambda p: p.requires_grad, self.actor.parameters()), actor_lr, momentum=actor_momentum, weight_decay=actor_wdecay)
        #self.critic_optim = optim.SGD(self.critic.parameters(), critic_lr, momentum=critic_momentum, weight_decay=critic_wdecay)
        actor_optim = optim.Adam(actor.parameters(), actor_lr,eps = 1e-5)
        critic_optim = optim.Adam(critic.parameters(), critic_lr,eps=1e-5)


        if use_gpu_model:
            T = lambda x: torch.cuda.FloatTensor(x)
        else:
            T = lambda x: torch.FloatTensor(x)
        rwd_list=[]      
        rwd_current = 0
        saved_model_path = None

        x_track=[]
        y_track=[]

        if(CL == True):
           CLScheduler = MotionBoundSchedule(2000, np.array([0.005, 0.005, 0.01, 0.01]))

        EPScheduler = EpisodeScheduler()

        if(not os.path.exists(log_dir)):
            os.mkdir(log_dir)
        writer = SummaryWriter(log_dir)
    
        for it in range(num_iter):
            #collect data
            #print("num of tasks:{}".format(self.vec_env.get_num_task()))
            start = time.time()
            data = runner.run()
            end = time.time()
            #print("collect data:{}".format(end-start))
            dataset = PPO_Dataset(data)

            atarg = dataset.advantage
            atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5) # trick: standardized advantage function

            adv_clip_rate = np.mean(np.abs(atarg) > 4)
            adv_max = np.max(atarg)
            adv_min = np.min(atarg)
            val_min = v_min;
            val_max = v_max;
            vtarg = dataset.vtarget
            vtarg_clip_rate = np.mean(np.logical_or(vtarg < val_min, vtarg > val_max))
            vtd_max = np.max(vtarg)
            vtd_min = np.min(vtarg)

            atarg = np.clip(atarg, -4, 4)
            vtarg = np.clip(vtarg, val_min, val_max)

            dataset.advantage = atarg
            dataset.vtarget = vtarg

            # logging interested variables
            N = np.clip(data["news"].sum(), a_min=1, a_max=None) # prevent divding 0
            avg_rwd = data["rwds"].sum()/N
            avg_step = data["samples"]/N
            rwd_per_step = avg_rwd / avg_step
            print("\n===== iter %d ====="% it)
            print("avg_rwd       = %f" % avg_rwd)
            print("avg_step      = %f" % avg_step)
            print("rwd_per_step  = %f" % rwd_per_step)

            writer.add_scalar('train/avg_rwd', avg_rwd, it)
            writer.add_scalar('train/avg_step', avg_step, it)
            writer.add_scalar('train/rwd_perstep', rwd_per_step, it)
            writer.add_scalar('train/time_limit', runner.env.env_method('get_episode_length', indices=0)[0], it)
            
        
            fail_rate = sum(data["fails"])/N
            total_sample += data["samples"]
        
            # special handling of bound channels
            # if(self.CL == True):
            #     flag_update = self.CLScheduler.update(avg_rwd)
            #     if(flag_update):
            #         bound_current = self.vec_env.get_motionbound()
            #         bound_current = bound_current - self.CLScheduler.delta_bound
            #         bound_current = np.clip(bound_current, np.array([0.001, 0.001, 0.001, 0.001]), np.array([10, 10, 10, 10]))
            #         self.vec_env.set_motionbound(bound_current)
    

            if (it % test_batch == 0):
             
                # test_step, test_rwd, test_rwd_perstep, test_vel, test_theta = runner.test()
                # print("test_rwd      = %f" % test_rwd)
                # print("test_step     = %f" % test_step)
                #importance sampling
                if(importance_sampling):
                    runner.env.env_method('set_save_state', True)
                    rwd_perstep = importance_sampler(runner)
                    runner.env.env_method('set_rwd_task', rwd_perstep.copy())
                    print("rwd per step:{}".format(rwd_perstep))
                    runner.env.env_method('set_save_state', False)
                else:
                    avg_step, avg_rwd,_,_,_ = runner.test()
                    rwd_perstep = avg_rwd/avg_step
                if(EPScheduler.update(it)):
                    #runner.env.set_perturb_prob(perturb_prob + 0.1)
                    episode_length = runner.env.env_method('get_episode_length', indices=0)[0]
                    runner.env.env_method('set_task_t', episode_length + 100)
                    #hack for faster training
                    #return model_path_final
                x_track.append(4096 * it)
                y_track.append(rwd_perstep)

                print("\n===== iter %d ====="% it)
                print("avg_rwd       = %f" % avg_rwd)
                print("rwd_per_step  = %f" % rwd_per_step)
                #print("test_rwd      = %f" % test_rwd)
                #print("test_step     = %f" % test_step)

            # start training
            pol_loss_avg    = 0
            symmetry_loss_avg = 0
            pol_surr_avg    = 0
            pol_abound_avg  = 0
            vf_loss_avg     = 0
            clip_rate_avg   = 0
            actor_grad_avg  = 0
            critic_grad_avg = 0
            gradient_norm = 0

            start = time.time()
            for epoch in range(epoch_size):
            #print("iter %d, epoch %d" % (it, epoch))
                for bit, batch in enumerate(dataset.batch_sample(batch_size)):
                    # prepare batch data
                    ob, ac, atarg, tdlamret, log_p_old = batch
                    ob = T(ob)
                    ac = T(ac)
                    atarg = T(atarg)
                    tdlamret = T(tdlamret).view(-1, 1)
                    log_p_old = T(log_p_old)

                    # clean optimizer cache
                    actor_optim.zero_grad()
                    critic_optim.zero_grad()

                    # calculate new log_pact
                    #ob_normed = s_norm(ob)
                    ob_normed = ob
                    m = actor.act_distribution(ob_normed)
                    vpred = critic(ob_normed)[:,0].reshape((-1,1))
                    log_pact = m.log_prob(ac)
                    if log_pact.dim() == 2:
                        log_pact = log_pact.sum(dim=1)


                    # PPO object, clip advantage object
                    ratio = torch.exp(log_pact - log_p_old)
                    surr1 = ratio * atarg
                    surr2 = torch.clamp(ratio, 1.0 - clip_threshold, 1.0 + clip_threshold) * atarg
                    pol_surr = -torch.mean(torch.min(surr1, surr2))
                    pol_loss = pol_surr 
                    pol_loss_avg += pol_loss.item()


                    # critic vpred loss
                    vf_criteria = nn.MSELoss()
                    vf_loss = vf_criteria(vpred, tdlamret) / (critic.v_std**2) # trick: normalize v loss
                

                    vf_loss_avg += vf_loss.item()

                    if (not np.isfinite(pol_loss.item())):
                        print("pol_loss infinite")
                        assert(False)
                        from IPython import embed; embed()

                    if (not np.isfinite(vf_loss.item())):
                         print("vf_loss infinite")
                         assert(False)
                         from IPython import embed; embed()
                    pol_loss.backward()
                    vf_loss.backward()

                    gradient_norm += calc_grad_norm(critic)
                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                    actor_optim.step()
                    critic_optim.step()
            end = time.time()
            #print("training time:{}".format(end-start))
            batch_num = (sample_size // batch_size)
            pol_loss_avg    /= batch_num
            vf_loss_avg     /= batch_num
            # save checkpoint
            if (checkpoint_batch > 0 and it % checkpoint_batch == 0):
                print("save check point to:{}".format(save_dir))
                actor.cpu()
                critic.cpu()
                s_norm.cpu()
                data = {"actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "s_norm": s_norm.state_dict()}
                if use_gpu_model:
                    actor.cuda()
                    critic.cuda()
                    s_norm.cuda()

                torch.save(data, "%s/checkpoint_%s.tar" % (save_dir, str(it)))
                model_path_final =  "%s/checkpoint_%s.tar" % (save_dir, str(it))
                #state_pool = vec_env.get_state_pool()
                #state_pool.save(save_dir + '/')

        return model_path_final


def loadModel(ckpt):
    s_norm, actor, critic,_,_ = load_model(ckpt)


  