'''
time based curriculum environment wrapper
'''
import numpy as np

class HandEnvWrapper(object):
    '''
    curriculum learning environment for locomotion tasks
    '''
    def __init__(self, env):
        self._env = env
        self._initial_state = self._env.reset()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.current_step = 0
        self._task_t = 1000 #standard settings for gym based tasks

    def step(self, a):
        state, reward, done, info = self._env.step(a)

        info['fail'] = done
        self.current_step +=1

        if (self.current_step > self._task_t or self._env.num_steps > self._env.episode_length):
            done = True

        if(self._env.save_state == True):
            phase = self._env.openloop_generator.phase_index(self._env.num_steps * self._env.frame_skip * self._env.time_step)
            if(info['rwd_chop1'] < 0.8 or info['rwd_chop2'] < 0.8 or info['pose_rwd']<0.8):
                sample = self._env.save_sample()
                self._env.sample_pool.update(sample, phase)

        return state, reward, done, info

    def reset(self):
        state = self._env.reset()
        self._initial_state = state
        self.current_step = 0
        return state

    def render(self, mode="human"):
        return self._env.render(mode)

    def get_task_t(self):
        return self._task_t

    def set_task_t(self, t):
        """ Set the max t an episode can have under training mode for curriculum learning
        """
        self._task_t = min(t, 4000)

    def set_task_idx(self, para):
        self._env.set_task_idx(para)

    def get_num_task(self):
        if self._env.mode == "single":
            return 1
        else:
            return len(self._env.env.openloop_generator_list)

    def set_sample_mode(self, mode):
        self._env.set_sample_mode(mode)

    def set_rwd_task(self,para):
        self._env.set_rwd_task(para)

    def get_episode_length(self):
        return np.min([self._env.episode_length, self._task_t])

    def set_save_state(self, flag = False):
        self._env.set_save_state(flag)

    def get_state_pool(self):
        return self._env.env.sample_pool

    def load_state(self, path, phase):
        self._env.sample_pool.load(path, phase)
