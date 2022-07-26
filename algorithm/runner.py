import torch
import numpy as np
from IPython import embed
import time
class GAERunner(object):
    '''
    Given environment, actor and critic, sample given number of samples
    '''



    def __init__(self, env, s_norm, actor, critic, sample_size, gamma, lam, 
          v_max, v_min, use_gpu_model=False):
        """
        Inputs:
            env       gym.env_vec, vectorized environment, need to have following funcs:
                        env.num_envs
                        env.observation_space
                        obs = env.reset()
                        obs, rwds, news, infos = env.step(acs)

            s_norm    torch.nn, normalizer for input states, need to have following funcs:
                        obst_normed = s_norm(obst)
                        s_norm.record(obt)
                        s_norm.update()

            actor     torch.nn, actor model, need to have following funcs:
                        m = actor.act_distribution(obst_norm), where m is a Gaussian distribution

            critic    torch.nn, critic model, need to have following funcs:
                        vpreds = critic(obst_norm)

            sample_size   int, number of samples per run()

            gamma     float, discount factor of reinforcement learning

            lam       float, lambda for GAE algorithm

            v_max     float, maximum of single step reward
            v_min     float, minimum of single step reward
        """
        self.env = env
        self.s_norm = s_norm
        self.actor = actor
        self.critic = critic
        self.nenv = nenv = env.num_envs
        self.obs = np.zeros((nenv, env.observation_space.shape[0]), dtype=np.float32)
        self.obs[:] = env.reset()
        self._use_gpu = use_gpu_model
        if use_gpu_model:
            self.toTorch = lambda x: torch.cuda.FloatTensor(x)
        else:
            self.toTorch = lambda x: torch.FloatTensor(x)
        self.sample_size = sample_size
        self.news = [True for _ in range(nenv)]

        # lambda used in GAE
        self.lam = lam
        # discount rate
        self.gamma = gamma

        self.v_max = v_max
        self.v_min = v_min

    def run(self):
        '''
        run policy and collect data for PPO training
        '''

        nsteps = int(self.sample_size/self.nenv) + 1

        mb_news = []
        mb_obs  = []
        mb_acs  = []
        mb_rwds = []
        mb_vpreds = []
        mb_alogps = []
        mb_ends   = []  # if the episode ends here
        mb_fails  = []  # if the episode ends here cause of failure
        mb_ob_ends= []
        mb_vends  = []

        self.end_rwds = []

        if self._use_gpu:
            self.s_norm.cpu()
            self.actor.cpu()
            self.critic.cpu()
        
        #self.s_norm.update()
        self.actor.update()
        self.critic.update()
        self.actor.set_mode('sample')
        self.critic.set_mode('sample')

        for _ in range(nsteps):
            #start = time.time()
            obst = torch.FloatTensor(self.obs)
            #self.s_norm.record(obst)
            #obst_norm = self.s_norm(obst)
            obst_norm = obst
            with torch.no_grad():
                # fully stocastic policy
                m = self.actor.act_distribution(obst_norm)
                acs = m.sample()
                alogps = torch.sum(m.log_prob(acs), dim=1).cpu().numpy()
                acs = acs.cpu().numpy()
            mb_news.append(self.news.copy())
            mb_obs.append(self.obs.copy())
            mb_acs.append(acs)
            #mb_vpreds.append(vpreds)
            mb_alogps.append(alogps)
            #start = time.time()
            self.obs[:], rwds, self.news, infos = self.env.step(acs)
            #end = time.time()
            #print("step time:{}".format(end - start))
            mb_rwds.append(rwds)
            fails = [infos[i]['fail'] for i in range(self.nenv)]
            mb_fails.append(fails)

            ends = np.zeros(self.nenv)
            for i, done in enumerate(self.news):
                if done:
                    ob_end = infos[i]["terminal_observation"]
                    mb_ob_ends.append(ob_end)
                    ends[i] = 1

            mb_ends.append(ends)
        if self._use_gpu:
          self.s_norm.cuda()
          self.actor.cuda()
          self.critic.cuda()

        mb_end_rwds     = np.array(self.end_rwds)
        mb_news = np.asarray(mb_news,   dtype=np.bool)
        mb_obs  = np.asarray(mb_obs,    dtype=np.float32)
        mb_acs  = np.asarray(mb_acs,    dtype=np.float32)
        mb_rwds = np.asarray(mb_rwds,   dtype=np.float32)
        mb_alogps=np.asarray(mb_alogps, dtype=np.float32)
        mb_fails= np.asarray(mb_fails,  dtype=np.bool)
        mb_ends = np.asarray(mb_ends,  dtype=np.bool)
        mb_ob_ends= np.asarray(mb_ob_ends, dtype=np.float32)

        with torch.no_grad():
            obst = self.toTorch(mb_obs)
            #obst_norm = self.s_norm(obst)
            obst_norm = obst
            mb_vpreds = self.critic(obst_norm)[:,:,0]
            #dim0, dim1, dim2 = mb_vpreds.shape
            #mb_vpreds = mb_vpreds.reshape(dim0, dim1)
            mb_vpreds = mb_vpreds.cpu().data.numpy()

            mb_vends = np.zeros(mb_ends.shape)
            if len(mb_ob_ends) > 0:
                obst = self.toTorch(mb_ob_ends)
                #obst_norm = self.s_norm(obst)
                obst_norm = obst
                vends = self.critic(obst_norm)[:,0]
                mb_vends[mb_ends] = vends.cpu().view(-1)
        with torch.no_grad():
            obst = self.toTorch(self.obs)
            #obst_norm = self.s_norm(obst)
            obst_norm = obst
            last_vpreds = self.critic(obst_norm).cpu()[:,0].view(-1).numpy()

            fail_end = np.logical_and(self.news, mb_fails[-1])
            succ_end = np.logical_and(self.news, np.logical_not(mb_fails[-1]))
            last_vpreds[fail_end] = self.v_min
            last_vpreds[succ_end] = mb_vends[-1][succ_end]

        mb_vtargs= np.zeros_like(mb_rwds)
        mb_advs  = np.zeros_like(mb_rwds)

        mb_nextvalues = mb_advs
        mb_nextvalues[:-1] = mb_vpreds[1:]
        fail_end = np.logical_and(mb_news[1:], mb_fails[:-1])
        succ_end = np.logical_and(mb_news[1:], np.logical_not(mb_fails[:-1]))
        mb_nextvalues[:-1][fail_end] = self.v_min
        mb_nextvalues[:-1][succ_end] = mb_vends[:-1][succ_end]
        mb_nextvalues[-1] = last_vpreds
        mb_delta = mb_advs
        mb_delta = mb_rwds + self.gamma * mb_nextvalues - mb_vpreds

        lastgaelam = 0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - self.news
            else:
                nextnonterminal = 1.0 - mb_news[t+1]
            mb_advs[t] = lastgaelam = mb_delta[t] + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_vtargs = mb_advs + mb_vpreds

        # save samples
        keys = ["news", "obs", "acs", "rwds", "fails","advs", "vtargs", "a_logps"]
        contents = map(sf01, (mb_news, mb_obs, mb_acs, mb_rwds, mb_fails, mb_advs, mb_vtargs, mb_alogps))

        data = {}
        for key, cont in zip(keys, contents):
            data[key] = cont
        data["samples"] = data["news"].size

        self.actor.set_mode('deter')
        self.critic.set_mode('deter')

        return data

    def test(self):
        """ Test current policy with unlimited timer

        Outputs:
            avg_step
            avg_rwd
        """
        self.actor.set_mode('sample')
        self.critic.set_mode('sample')

         #save good states

        alive = np.array([True for _ in range(self.nenv)])
        any_alive = True
        acc_rwd = np.zeros(self.nenv)
        acc_step = np.zeros(self.nenv)
        acc_vel = np.zeros(self.nenv)
        acc_energy = np.zeros(self.nenv)
        acc_theta = np.zeros(self.nenv)

        self.obs = self.env.reset()
        self.news = [True for _ in range(self.nenv)]
        if self._use_gpu:
            self.s_norm.cpu()
            self.actor.cpu()
        while any_alive:
            # normalize input state before feed to actor & critic
            obst = torch.FloatTensor(self.obs)
            self.s_norm.record(obst)
            #obst_norm = self.s_norm(obst)
            obst_norm = obst

            with torch.no_grad():
                # with probability exp_rate to act stochastically
                acs = self.actor.act_deterministic(obst_norm)
                acs = acs.cpu().numpy()

            self.obs[:], rwds, self.news, infos = self.env.step(acs)

            # decide which are alive, since timer is set to max, so using self.news as fails
            alive = np.logical_and(alive, np.logical_not(self.news))
            energy = np.array([info["energy"] for info in infos]) if "energy" in infos[0] else np.array([-1 for _ in infos])
            vel = np.array([info["vel"] for info in infos]) if "vel" in infos[0] else np.array([-1 for _ in infos])
            theta = np.array([info["theta"] for info in infos]) if "theta" in infos[0] else np.array([-1 for _ in infos])
            # record the rwd and step for alive agents
            acc_rwd += rwds * alive
            acc_energy += energy*alive
            acc_vel += vel*alive
            acc_step += alive
            acc_theta += abs(theta) * alive

            # decide if any are alive
            any_alive = np.any(alive)

        if self._use_gpu:
            self.s_norm.cuda()
            self.actor.cuda()

        avg_step = np.mean(acc_step)
        avg_rwd  = np.mean(acc_rwd)
        avg_rwd_perstep = np.mean(acc_rwd/(acc_step+1))
        avg_theta = np.mean(acc_theta)/self.env.env_method('get_task_t', indices=0)[0]
        avg_energy = np.mean(acc_energy)/self.env.env_method('get_task_t', indices=0)[0]
        avg_vel = np.mean(acc_vel)/self.env.env_method('get_task_t', indices=0)[0]

        self.obs = self.env.reset()

        self.actor.set_mode('deter')
        self.critic.set_mode('deter')


        return avg_step, avg_rwd, avg_rwd_perstep, avg_vel, avg_theta
        
class SACRunner(object):
    '''
    Given environment, actor, sample given number of samples for Soft-Actor Critic algorithm
    '''



    def __init__(self, env, actor, replay_buffer, sample_size):
        """
        Inputs:
            env       gym.env_vec, vectorized environment, need to have following funcs:
                        env.num_envs
                        env.observation_space
                        obs = env.reset()
                        obs, rwds, news, infos = env.step(acs)
            actor     actor policy network using a tanhGaussian structure
            sample_size   int, number of samples per run()
        """
        self.env = env
        self.actor = actor
        self.nenv = nenv = env.num_envs
        self.obs = np.zeros((nenv, env.observation_space.shape[0]), dtype=np.float32)
        self.obs[:] = env.reset()
        self.sample_size = sample_size
        self.news = [True for _ in range(nenv)]
        self.reply_buffer = replay_buffer

    def run(self):
        '''
        run policy and collect data for SAC training
        '''
        nsteps = int(self.sample_size/self.nenv) + 1

        self.end_rwds = []

        for _ in range(nsteps):
            state = self.obs.copy()
            acs, _ = self.actor.get_action(state)
            self.obs[:], rwds, self.news, infos = self.env.step(acs)
            fails = [infos[i]['fail'] for i in range(self.nenv)]
            ends = np.zeros(self.nenv)
            for i, done in enumerate(self.news):
                if done:
                    ob_end = infos[i]["terminal_observation"]
                    ends[i] = 1
                    self.reply_buffer.add_sample(observation=state[i], action=acs[i], reward=np.array(rwds[i]), next_observation=ob_end,
                           terminal=np.array([fails[i]]), env_info={})
                else:
                    self.reply_buffer.add_sample(observation=state[i], action=acs[i], reward=np.array(rwds[i]), next_observation=self.obs.copy()[i],
                           terminal=np.array([fails[i]]), env_info={})
                
        self.reply_buffer.terminate_episode()

    def test(self, states_buffer=None):
        """ Test current policy with unlimited timer

        Outputs:
            avg_step
            avg_rwd
        """
        alive = np.array([True for _ in range(self.nenv)])
        any_alive = True
        acc_rwd = np.zeros(self.nenv)
        acc_step = np.zeros(self.nenv)
        #acc_vel = np.zeros(self.nenv)
        #acc_energy = np.zeros(self.nenv)
        #acc_theta = np.zeros(self.nenv)

        self.obs = self.env.reset()
        self.news = [True for _ in range(self.nenv)]
        while any_alive:
            # normalize input state before feed to actor & critic
            state = self.obs.copy()
            acs, _ = self.actor.get_action(state, deterministic=True)
            self.obs[:], rwds, self.news, infos = self.env.step(acs)

            # decide which are alive, since timer is set to max, so using self.news as fails
            alive = np.logical_and(alive, np.logical_not(self.news))
            # record the rwd and step for alive agents
            acc_rwd += rwds * alive
            acc_step += alive
            any_alive = np.any(alive)

            if(states_buffer!=None):
                pass

        avg_step = np.mean(acc_step)
        avg_rwd  = np.mean(acc_rwd)
        self.obs = self.env.reset()

        return avg_step, avg_rwd  



                    



def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])





 
        

      

      