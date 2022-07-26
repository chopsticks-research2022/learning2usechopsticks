import copy

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.utils import seeding


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path):

        model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(model)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self._set_action_space()

        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')


    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Step the environment
        """
        raise NotImplementedError()

    def reset(self):
        obs = self._reset_sim()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=500, height=500):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _set_action_space(self):
        bounds = np.float32(self.sim.model.actuator_ctrlrange.copy())
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high, dtype='float32')

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and return its observation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _env_setup(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass