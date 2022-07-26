import json
import numpy as np
import glm

from utils.convert_util import *


def lerp(x1, x2, a):
    '''
    linear interpolate x1, x2 given the interpolation factor a
    '''
    return (1 - a) * x1 + a * x2


def slerp(q1, q2, a):
    '''
    spherical linear interpolation of two quaternions q1, q2 given the interpolation factor a
    '''
    return convert_glm2quat(glm.slerp(convert_quat2glm(q1), convert_quat2glm(q2), a))


class OpenloopGenerator():

    def __init__(self, openloop_file):
        
        # load information from saved trajectories
        if(type(openloop_file) == str):
            with open(openloop_file, 'r') as f:
                motion_dict = json.load(f)
        else:
            motion_dict = openloop_file

        for key in motion_dict.keys():
            if(type(motion_dict[key]) == list):
                motion_dict[key] = np.array(motion_dict[key])
        self.motion=motion_dict["openloop_arm"]
        self.vels = motion_dict["vel_openloop_arm"][:,1:]

        self.motion_chopsticks = motion_dict['motion_chopsticks'][:,1:]
        self.vels_chopsticks = motion_dict['vel_chopsticks'][:,1:]
        self.geom_size = motion_dict['geom_size']

        if('contact' in motion_dict.keys()):
            self.q_init = np.array(motion_dict['q_init'])
            self.contact_mask = motion_dict['contact']
            self.phase = motion_dict['phase']
            self.object_idx = motion_dict['object_idx']

            self.phase0_index = []
            self.time_object = [[] for i in range(self.object_idx[-1] + 1)]

            for i in range(self.phase.shape[0]):
                    if(self.phase[i] == 0 and self.phase[(i+20)%self.phase.shape[0]] == 0):
                        self.time_object[int(np.floor(self.object_idx[i]))].append(i)
            self.num_objects = len(self.time_object)
        if('motion_object' in motion_dict.keys()):
            self.object_motion = motion_dict['motion_object'][:,1:]
            self.object_vel = motion_dict['vel_object'][:,1:] 
        if('openloop_motion' in motion_dict.keys()):
            self.openloop_motion = motion_dict['openloop_motion'][:,1:]
            self.vels_openloop = motion_dict['vel_openloop'][:,1:] 
        if('rel_pos' in motion_dict.keys()):
            self.rel_pos = motion_dict['rel_pos']
            self.dr_chopstick2 = convert_pos2glm(np.array(motion_dict['dr_chopstick2']))
            self.dq_chopstick2 = convert_quat2glm(np.array(motion_dict['dq_chopstick2']))
            self.grasp_mode = np.array(motion_dict['grasp_mode'])
            self.tip_pos = np.array(motion_dict['tip_pos']).reshape((-1,3))
            self.qpos = np.array(motion_dict['qpos'])
        else:
            self.rel_pos = 0
        if('plate_pos' in motion_dict.keys()):
            self.plate_pos = np.array(motion_dict['plate_pos'])
            self.plate_size = np.array(motion_dict['plate_size'])
        else:
            self.plate_pos = -1
        self.dt = self.motion[0, 0]
        self.motion_time = self.dt*(self.motion.shape[0]-1)
       
        self.acc = np.zeros((self.vels.shape[0], 7))
        self.acc[:-1,:] = (self.vels[1:,] - self.vels[:-1,:])/self.dt
        self.acc[-1,:] = self.acc[-2, :].copy()
        self.openloop_arm = self.motion[:,1:].copy()


    def _compute_interp_coeff(self, t):
        '''
        get the time idx range (t1, t2) and the linear interp coefficient (alpha)
        '''
        t1 = int(t / self.dt)
        t2 = t1 + 1
        a = abs(t - t1 * self.dt) / self.dt
        return t1, t2, a


    def chopsticks(self, t):
        '''
        compute the motion of the chopsticks at time t
        '''
        if abs(t - self.motion_time) < 0.0001 or t > self.motion_time:
            return self.motion_chopsticks[-1], self.vels_chopsticks[-1]

        t1, t2, a = self._compute_interp_coeff(t)
        blend_motion = np.zeros(8)  # (chop_pos + chop_quat + angle_between_chop1_and_chop2)

        blend_motion[0:3] = lerp(self.motion_chopsticks[t1, 0:3], self.motion_chopsticks[t2, 0:3], a)   # chop_pos
        blend_motion[3:7] = slerp(self.motion_chopsticks[t1, 3:7], self.motion_chopsticks[t2, 3:7], a)  # chop_quat
        blend_motion[-1] = lerp(self.motion_chopsticks[t1, -1], self.motion_chopsticks[t2, -1], a)      # angle_between_chop1_and_chop2
        blend_vel = lerp(self.vels_chopsticks[t1], self.vels_chopsticks[t2], a)
        
        return blend_motion, blend_vel


    def openloop(self,t):
        '''
        compute the openloop of the arm
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return self.openloop_arm[-1], self.vels[-1]
        t1, t2, a = self._compute_interp_coeff(t)
        blend_motion = lerp(self.openloop_arm[t1], self.openloop_arm[t2], a)
        blend_vel = lerp(self.vels[t1], self.vels[t2], a)
        return blend_motion, blend_vel


    def acc_arm(self,t):
        '''
        compute the acc of the arm
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return self.acc[-1]
        t1, t2, a = self._compute_interp_coeff(t)
        return lerp(self.acc[t1], self.acc[t2], a)


    def openloop_full(self,t):
        '''
        compute the openloop of the whole robot
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return self.openloop_motion[-1], self.vels_openloop[-1]
        t1, t2, a = self._compute_interp_coeff(t)
        blend_motion = lerp(self.openloop_motion[t1], self.openloop_motion[t2], a)
        blend_vel = lerp(self.vels_openloop[t1], self.vels_openloop[t2], a)
        return blend_motion, blend_vel


    def object(self, t):
        '''
        compute the motion of the object
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return self.object_motion[-1], self.object_vel[-1]
        
        t1, t2, a = self._compute_interp_coeff(t)
        q0, q1 = self.object_motion[t1], self.object_motion[t2]

        blend_motion_object = np.zeros(7)   # (pos + quat)
        blend_motion_object[0:3] = lerp(q0[0:3], q1[0:3], a)    # pos
        blend_motion_object[3:7] = slerp(q0[3:7], q1[3:7], a)   # quat
        blend_vel_object = lerp(self.object_vel[t1], self.object_vel[t2], a)

        return blend_motion_object, blend_vel_object
    

    def contact(self, t):
        '''
        return the contact mask: whether the object is in contact with the chopsticks
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return self.contact_mask[-1]
        t1, t2, a = self._compute_interp_coeff(t)
        return lerp(self.contact_mask[t1], self.contact_mask[t2], a)


    def phase_index(self, t):
        '''
        return the phase: which phase: pregrasp, lift, move or release
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return self.phase[-1]
        t1, t2, a = self._compute_interp_coeff(t)
        return int(np.round(lerp(self.phase[t1], self.phase[t2], a)))


    def object_idx_index(self, t):
        '''
        return the object_idx: which object is manipulated
        '''
        if abs(t - self.motion_time) < 1e-4 or t > self.motion_time:
            return int(self.object_idx[-1])
        t1, t2, a = self._compute_interp_coeff(t)
        return int(np.round(lerp(self.object_idx[t1], self.object_idx[t2], a)))     # the object index should be integar


    def sample_time(self, object_id=None, phase=0):
        '''
        given phase, return a time index that correpsonds to the phase
        '''
        num_objects = len(self.time_object)
        if object_id is None:
            object_id = np.random.randint(0, num_objects)
        idx = np.random.randint(0, len(self.time_object[object_id]))
        return self.time_object[object_id][idx] * self.dt
