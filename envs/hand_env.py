import os

import glm
import numpy as np
from gym import utils, spaces
from mujoco_py import functions as mjf
from mujoco_py import MjSim, load_model_from_path
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation as R

from algorithm.statepool import StatePool
from envs.motiongenerator import OpenloopGenerator
from envs.mujoco_env import MujocoEnv
from utils.convert_util import *


def find_index(prob_array, rand_num):
    # given the probility array and generated random number, output the index of the task
    prob_array_accum = prob_array.copy()
    for i in range(prob_array.shape[0]):
        prob_array_accum[i] = prob_array[0: i + 1].sum()
    for i in range(len(prob_array_accum)):
        if(rand_num < prob_array_accum[i]):
            return i


class HandEnv(MujocoEnv, utils.EzPickle):

    def __init__(self, model_path, openloop_path, mode='single', render_mode=None, debug=False):
        self.mode = mode
        self.render_mode = render_mode

        self.frame_skip = 1
        self.num_steps = 0
        self.num_steps_sim = 0
        self.time_step = 0.01
        self.object_dof = 7 + 7
        self.mocap_dof = 8
        self.save_state = False     # whether to save the simulated state
        self.sample_pool = StatePool(4, 200)

        self.object_idx = 0
        self.sample_mode = "deter"

        #set prerequsite for hand control:openloop, motion file and hand pose
        if(self.mode == 'single'):
            self.openloop_path = openloop_path
            self.openloop_generator = OpenloopGenerator(self.openloop_path)

        elif(self.mode == 'batch'):
            self.openloop_idx = 0
            files = os.listdir(openloop_path)
            self.openloop_generator_list = []
            for i in range(len(files)):
                self.openloop_generator_list.append(OpenloopGenerator(openloop_path + '/' + files[i]))
            self.openloop_generator = self.openloop_generator_list[self.openloop_idx]

            # reward for each task
            self.rwd_task = np.zeros(len(self.openloop_generator_list))

        else:
            raise NotImplementedError

        self.dr_chopstick2 = self.openloop_generator.dr_chopstick2
        self.dq_chopstick2 = self.openloop_generator.dq_chopstick2

        self.episode_length = np.around(self.openloop_generator.motion_time/(self.time_step * self.frame_skip))
        
        #parameters of the spd controller
        self.num_internal_joints = 29
        self.dof_control = 4 + 3 + self.num_internal_joints
        self.dof_hand_chopstick = self.dof_control + 14 + self.mocap_dof
        self.jkp= np.array([5000]*4 + [5000]*3 + [1]*self.num_internal_joints)
        self.jkd = np.array([500]*4 + [500]*3 + [0.1]*self.num_internal_joints)
        self.force_limits = np.array([1000]*4 + [1000]*3 + [100]*self.num_internal_joints)
        self.dr_rotaxis1 = np.array([0,0,0])
        self.dr_rotaxis2 = np.array([0,0,0])

        chop_xml = './data/chopsticks_xml/arm.xml'
        self.sim_chopsticks = MjSim(load_model_from_path(chop_xml))

        MujocoEnv.__init__(self, model_path)
        utils.EzPickle.__init__(self)


    def _init_env(self):
        self.geom_index = self.sim.model.geom_name2id("C_ff")
        self.geom_middle = self.sim.model.geom_name2id("C_mf")
        self.geom_thumb = self.sim.model.geom_name2id("C_th")
        self.geom_ring = self.sim.model.geom_name2id("C_rf")
        self.geom_little = self.sim.model.geom_name2id("C_lf")
        self.geom_thumb_middle = self.sim.model.geom_name2id("C_thm")
        self.geom_ffproxmial = self.sim.model.geom_name2id("C_ffproximal_trick")
        self.geom_index_contact = self.sim.model.geom_name2id("C_ffdistal")
        self.geom_middle_contact = self.sim.model.geom_name2id("C_mfdistal")
        self.geom_thumb_contact = self.sim.model.geom_name2id("C_thdistal")
        self.geom_ring_contact = self.sim.model.geom_name2id("C_rfdistal")
        self.geom_little_contact = self.sim.model.geom_name2id("C_lfdistal")
        self.geom_thumb_middle_contact = self.sim.model.geom_name2id("C_thmiddle")

        self.geom_chopstick1 = self.sim.model.geom_name2id("chopstick1")
        self.geom_chopstick2 = self.sim.model.geom_name2id("chopstick2")
        self.geom_chopstick1_end = self.sim.model.geom_name2id("chopstick1_end")
        self.geom_chopstick2_end = self.sim.model.geom_name2id("chopstick2_end")

        self.site_end1 = self.sim.model.site_name2id("end1")
        self.site_end2 = self.sim.model.site_name2id("end2")
        self.site_end1_v = self.sim.model.site_name2id("end1_v")
        self.site_end2_v = self.sim.model.site_name2id("end2_v")
        self.site_joint = self.sim.model.site_name2id("c_joint")

        self.body_chopstick1 = self.sim.model.body_name2id("chopstick1")
        self.body_chopstick2 = self.sim.model.body_name2id("chopstick2")
        self.body_chopstick2_v = self.sim.model.body_name2id("chopstick2_v")
        self.body_palm = self.sim.model.body_name2id("palm")

        self.body_object = self.sim.model.body_name2id("object_" + str(self.object_idx))
        self.geom_object = self.sim.model.geom_name2id("object_" + str(self.object_idx))

        self._construct_tipdict()


    def _env_setup(self):
        self._init_env()

        # initialize pose parameters: pos and rotation
        start_t = 0
        pose_arm, vel_arm = self.openloop_generator.openloop(start_t)
        num_objects_traj = self.openloop_generator.num_objects
        self.sim.model.geom_size[-num_objects_traj: ] = self.openloop_generator.geom_size.copy()
        self.sim.data.qpos[0:7] = pose_arm
        self.sim.data.qvel[0:7] = vel_arm
        self.sim.data.qpos[7:self.dof_control + 0] = self.standard_pose[7: self.dof_control + 0]
        shape = self.openloop_generator.q_init[8:].shape[0]
    
        self.sim.data.qpos[self.dof_hand_chopstick: self.dof_hand_chopstick + shape] = self.openloop_generator.q_init[8:]
        r_chop2, q_chop2, r_chop1, q_chop1 = self._chop_kin(start_t)
        self.sim.data.qpos[self.dof_control: self.dof_control + 3] = r_chop1
        self.sim.data.qpos[self.dof_control + 3: self.dof_control + 7] = q_chop1
        self.sim.data.qpos[self.dof_control + 7: self.dof_control + 10] = r_chop2
        self.sim.data.qpos[self.dof_control + 10: self.dof_control + 14] = q_chop2
        self.sim.data.ctrl[:] = 0 #self.init_ctrl.copy()
        self.sim.data.qacc_warmstart[:] = 0
        self.sim.forward()

        state = self.save_sample()
        self.sample_pool.save_init(state)

        quat_palm = self.sim.data.body_xquat[self.body_palm]
        quat_palm = glm.quat(quat_palm[0], quat_palm[1], quat_palm[2], quat_palm[3])
        quat_chopstick2 = self.sim.data.body_xquat[self.body_chopstick2]
        quat_chopstick2 = glm.quat(quat_chopstick2[0], quat_chopstick2[1], quat_chopstick2[2], quat_chopstick2[3])

        dr = self.sim.data.body_xpos[self.body_chopstick2] - self.sim.data.body_xpos[self.body_palm]
        dr = glm.quat(0, dr[0], dr[1], dr[2])
        self.dr_rotaxis1, self.dr_rotaxis2 = self._get_rot_axis(0)


    def _set_action_space(self):
        jnt_range = np.float32(self.sim.model.jnt_range.copy())
        bounds = jnt_range[:self.dof_control]
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high, dtype='float32')
    

    def _construct_tipdict(self):
        self.grasp_mode = self.openloop_generator.grasp_mode
        self.tip_dict = {}
        self.tip_dict['thumb'] = self.sim.model.geom_name2id('chopstick1')
        tip_name = ['index', 'middle', 'ring', 'little']
        for i in range(len(tip_name)):
            if(self.grasp_mode[i] == 1):
                self.tip_dict[tip_name[i]] = self.sim.model.geom_name2id('chopstick1')
            elif(self.grasp_mode[i] == 2):
                self.tip_dict[tip_name[i]] = self.sim.model.geom_name2id('chopstick2')
            else:
                self.tip_dict[tip_name[i]] = -1
        self.tip_pos = self.openloop_generator.tip_pos.copy()
        self.standard_pose = self.openloop_generator.qpos.copy()


    def save_sample(self):
        #save the simulation state
        qpos = self.sim.data.qpos[:].copy()
        qvel = self.sim.data.qvel[:].copy()
        init_ctrl = self.sim.data.ctrl[:].copy()
        qacc = self.sim.data.qacc_warmstart[:].copy()
        state = {}
        state['qpos'] = qpos
        state['qvel'] = qvel
        state['ctrl'] = init_ctrl
        state['qacc'] = qacc
        state['num_steps'] = self.num_steps
        state['object_idx'] = self.openloop_generator.object_idx_index(self.num_steps * self.frame_skip * self.time_step)
        state['num_steps_sim'] = self.num_steps_sim
        state['dr_rotaxis1'] = self.dr_rotaxis1.copy()
        state['dr_rotaxis2'] = self.dr_rotaxis2.copy()
        if(self.mode == 'batch'):
            state['openloop_idx'] = self.openloop_idx
        return state


    def set_task_idx(self, idx):
        self.openloop_idx = idx

    def set_sample_mode(self, mode):
        self.sample_mode = mode

    def set_rwd_task(self, rwd_task):
        self.rwd_task = rwd_task

    def set_save_state(self, save_state):
        self.save_state = save_state


    # Reward methods
    # ----------------------------

    def _compute_rwd_hand(self):
        pos_arm, _ = self.openloop_generator.openloop(self.t)
        rwd_arm = np.exp(-20 * np.linalg.norm(self.sim.data.qpos[0:7] - pos_arm))
        
        joint_cmc = np.concatenate([self.sim.data.qpos[7:9], self.sim.data.qpos[13:15], self.sim.data.qpos[19:21], self.sim.data.qpos[30:32]])
        joint_ref = np.concatenate([self.standard_pose[7:9], self.standard_pose[13:15], self.standard_pose[19:21], self.standard_pose[30:32]])
        rwd_hand = np.exp(-1 * np.linalg.norm(joint_cmc - joint_ref))

        return rwd_hand * rwd_arm


    def _compute_rwd_chopsticks(self):
        dr_chop2, dq_chop2, dr_chop1, dq_chop1 = self._get_chopsticks_drdq()

        dq2 = glm.inverse(self.dq_chopstick2) * dq_chop2
        dtheta2 = 2 * np.arccos(np.clip(dq2[3],-1,1))

        dr_sim_chop1_pivot, dr_sim_chop2_pivot = self._get_rot_axis(self.t)
        dr1 = np.linalg.norm(self.dr_rotaxis1 - dr_sim_chop1_pivot)
        dr2 = np.linalg.norm(self.dr_rotaxis2 - dr_sim_chop2_pivot)
        
        pos_end1 = self.sim.data.site_xpos[self.site_end1]
        pos_end1_v = self.sim.data.site_xpos[self.site_end1_v]

        pos_chop2 = self.sim.data.body_xpos[self.body_chopstick2]
        quat_chop2 = self.sim.data.body_xquat[self.body_chopstick2]
        quat_chop2 = R.from_quat([quat_chop2[1], quat_chop2[2], quat_chop2[3], quat_chop2[0]])
        up_vec = quat_chop2.apply(np.array([0,0,1]))
        pos_chop1_site = self.sim.data.site_xpos[self.site_end1].copy()
        pos_chop1_site = pos_chop1_site - pos_chop2
        pos_chop1_site = abs(np.dot(pos_chop1_site, up_vec))

        if(self.openloop_generator.phase_index(self.t) != 0):
            w_phase = 20
        else:
            w_phase = 40

        rwd_chopstick1 = np.exp(-10 * np.linalg.norm(pos_end1 - pos_end1_v) - w_phase * dr1 - 20 * pos_chop1_site)
        rwd_chopstick2 = np.exp(-10 * abs(dtheta2) - 40 * dr2)

        return rwd_chopstick1, rwd_chopstick2


    def _compute_rwd_object(self):
        pose_object, _ = self.openloop_generator.object(self.t)

        if(self.openloop_generator.phase_index(self.t) != 3):
            rwd_object = np.exp(-20*np.linalg.norm(pose_object[0:3] - self.sim.data.body_xpos[self.body_object]))
        else:
            rwd_object = 1

        return rwd_object


    def _compute_rwd_contact_and_collision(self):
        dis_contact_thumb, dis_contact_index, dis_contact_middle, dis_contact_ring, dis_contact_little, dis_end12object, dis_end22object = self._compute_dis(self.t)
        contact_mask = self.openloop_generator.contact(self.t)

        rwd_contact = np.exp(-20 * (dis_contact_index + dis_contact_middle + dis_contact_ring + dis_contact_little + dis_contact_thumb))

        if(contact_mask >= 0.5):
            rwd_collision = 0.1 * np.exp(-40 * np.max([-0.001, dis_end22object]) - 40 * np.max([-0.001, dis_end12object]))
        else:
            rwd_collision = 0

        return rwd_contact, rwd_collision


    # Computation methods
    # ----------------------------

    def _compute_desired_accel(self, qpos_err, qvel_err):
        dt = self.sim.model.opt.timestep
        nv = self.sim.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.sim.model, M, self.sim.data.qM)
        M.resize(self.sim.model.nv, self.sim.model.nv)
        M = M[0:self.dof_control, 0:self.dof_control]
        C = self.sim.data.qfrc_bias.copy()[0:self.dof_control]
        K_p = np.diag(self.jkp)
        K_d = np.diag(self.jkd)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()


    def _compute_torque(self, target_pos, target_vel):
        '''
        Compute the torque using pd-controller
        '''
        dt = self.sim.model.opt.timestep
        qpos = self.sim.data.qpos.copy()[0:self.dof_control]
        qvel = self.sim.data.qvel.copy()[0:self.dof_control]

        qpos_err = (qpos + qvel*dt) - target_pos
        qvel_err = qvel - target_vel
        q_accel = self._compute_desired_accel(qpos_err, qvel_err)
        qvel_err += q_accel * dt
        torque = -self.jkp * qpos_err[:] - self.jkd * qvel_err[:]
        torque = np.clip(torque, -self.force_limits, self.force_limits)
        return torque


    def _get_chopsticks_drdq(self):
        '''
        Compute the chopsticks' position and quaternion
        '''
        quat_palm = self.sim.data.body_xquat[self.body_palm]
        quat_palm = glm.quat(quat_palm[0], quat_palm[1], quat_palm[2], quat_palm[3])
        quat_chopstick2 = self.sim.data.body_xquat[self.body_chopstick2]
        quat_chopstick2 = glm.quat(quat_chopstick2[0], quat_chopstick2[1], quat_chopstick2[2], quat_chopstick2[3])
        dq_chopstick2 = glm.inverse(quat_palm) * quat_chopstick2

        dr2 = self.sim.data.body_xpos[self.body_chopstick2] - self.sim.data.body_xpos[self.body_palm]
        dr2 = glm.quat(0, dr2[0], dr2[1], dr2[2])
        dr_chopstick2 = glm.inverse(quat_palm) * dr2 * quat_palm

        quat_chopstick1 = convert_quat2glm(self.sim.data.body_xquat[self.body_chopstick1])
        dq_chopstick1 = glm.inverse(quat_palm) * quat_chopstick1
        dr1 = self.sim.data.site_xpos[self.site_joint] - self.sim.data.body_xpos[self.body_palm]
        dr1 = glm.quat(0, dr1[0], dr1[1], dr1[2])
        dr_chopstick1 = glm.inverse(quat_palm) * dr1 * quat_palm
        return dr_chopstick2, dq_chopstick2, dr_chopstick1, dq_chopstick1


    def _get_rot_axis(self, t):
        '''
        Compute the chopsticks' pivot point at time t
        '''
        pos_site_thumb, pos_site_thumb_middle, _,_,_,_= self._compute_tippos(t)
        quat_palm = convert_quat2glm(self.sim.data.body_xquat[self.sim.model.body_name2id('palm')])
        dr2 = pos_site_thumb_middle - self.sim.data.body_xpos[self.sim.model.body_name2id('palm')]
        dr2 = glm.quat(0, dr2[0], dr2[1], dr2[2])
        dr2 = glm.inverse(quat_palm) * dr2 * quat_palm
        dr2 = convert_glm2pos(dr2)

        dr1 = pos_site_thumb - self.sim.data.body_xpos[self.sim.model.body_name2id('palm')]
        dr1 = glm.quat(0, dr1[0], dr1[1], dr1[2])
        dr1 = glm.inverse(quat_palm) * dr1 * quat_palm
        dr1 = convert_glm2pos(dr1)
        return dr1, dr2


    def _compute_tippos(self, t):
        '''
        compute the positions of tips of thumb, thumb_site, index, middle, ring, little
        '''
        r_chop2, q_chop2, r_chop1, q_chop1 = self._chop_kin(t)
        rel_pos = self.openloop_generator.tip_pos.copy()# + self.openloop_generator.rel_pos#rel positions on the chops
        rel_pos[:,0] += self.openloop_generator.rel_pos
        output = []
        chop_id  = [1,2] + self.grasp_mode.tolist()
        for i in range(6):
            if(chop_id[i] == 1):
                pos = convert_quat2glm(q_chop1) * convert_pos2glm(rel_pos[i]) * glm.inverse(convert_quat2glm(q_chop1))
                pos = r_chop1 + convert_glm2pos(pos)
            elif(chop_id[i] == 2):
                pos = convert_quat2glm(q_chop2) * convert_pos2glm(rel_pos[i]) * glm.inverse(convert_quat2glm(q_chop2))
                pos = r_chop2 + convert_glm2pos(pos)
            else:
                pos = np.array([0,0,0])
            output.append(pos.copy())
        return output[0], output[1], output[2],output[3], output[4], output[5]


    def _compute_dis(self, t):
        '''
        Compute the distances between the finger tips from its planned position on the chopsticks,
        also the distance of the object with the chopsticks tip.
        '''
        dis_contact_thumb = 0
        dis_contact_index = 0
        dis_contact_middle = 0
        dis_contact_ring = 0
        dis_contact_little = 0
        dis_end12object = 0
        dis_end22object = 0
        
        pos_site_thumb, _, pos_site_index, pos_site_middle,pos_site_ring, pos_site_little = self._compute_tippos(t)
        
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            # compute distances between fingers with chopsticks
            if((contact.geom2 == self.geom_chopstick1 and contact.geom1 == self.geom_thumb) or (contact.geom1 == self.geom_chopstick1 and contact.geom2 == self.geom_thumb)):
                contact_point_thumb = contact.pos
                dis_contact_thumb = np.linalg.norm(contact_point_thumb - pos_site_thumb)

            if((contact.geom2 == self.tip_dict['index'] and contact.geom1 == self.geom_index) or (contact.geom1 == self.tip_dict['index'] and contact.geom2 == self.geom_index)):
                if(self.grasp_mode[0] != 0):
                    contact_point_index = contact.pos
                    dis_contact_index = np.linalg.norm(contact_point_index - pos_site_index)
            
            if((contact.geom2 == self.tip_dict['middle'] and contact.geom1 == self.geom_middle) or (contact.geom1 == self.tip_dict['middle'] and contact.geom2 == self.geom_middle)):
                if(self.grasp_mode[1] != 0):
                    contact_point_middle = contact.pos
                    dis_contact_middle = np.linalg.norm(contact_point_middle - pos_site_middle)

            if((contact.geom2 == self.tip_dict['ring'] and contact.geom1 == self.geom_ring) or (contact.geom1 == self.tip_dict['ring'] and contact.geom2 == self.geom_ring)):
                if(self.grasp_mode[2] != 0):
                    contact_point_ring = contact.pos
                    dis_contact_ring = np.linalg.norm(contact_point_ring - pos_site_ring)
            
            if((contact.geom2 == self.tip_dict['little'] and contact.geom1 == self.geom_little) or (contact.geom1 == self.tip_dict['little'] and contact.geom2 == self.geom_little)):
                if(self.grasp_mode[3] != 0):
                    contact_point_little = contact.pos
                    dis_contact_little = np.linalg.norm(contact_point_little - pos_site_little)

            # compute distances between chopsticks with object
            if((contact.geom2 == self.geom_chopstick2_end and contact.geom1 == self.geom_object) or (contact.geom1 == self.geom_chopstick2_end and contact.geom2 == self.geom_object)):
                dis_end22object = contact.dist 
            
            if((contact.geom2 == self.geom_chopstick1_end and contact.geom1 == self.geom_object) or (contact.geom1 == self.geom_chopstick1_end and contact.geom2 == self.geom_object)):
                dis_end12object = contact.dist        
        
        return dis_contact_thumb, dis_contact_index, dis_contact_middle, dis_contact_ring, dis_contact_little, dis_end12object, dis_end22object
    
    
    def _compute_kinematics(self, t):
        '''
        Compute the transformation of the chopsticks in global coordinate system
        '''
        pose, vel = self.openloop_generator.chopsticks(t)
        pos2 = pose[0:3]
        quat2 = glm.quat(pose[3], pose[4], pose[5], pose[6])
        theta = pose[-1]
        dq = glm.quat(np.cos(theta/2), 0, 0, np.sin(theta/2))
        q = quat2 * dq
        dr = quat2 * glm.quat(0, 0.045, 0.035, 0) * glm.inverse(quat2) + q*glm.quat(0, -0.045, 0, 0)*glm.inverse(q)
        dr = np.array([dr[0], dr[1], dr[2]])
        pos1 = pos2 + dr 
        quat1 = quat2*dq
        quat1 = np.array([quat1[3], quat1[0], quat1[1], quat1[2]])
        quat2 = np.array([quat2[3], quat2[0], quat2[1], quat2[2]])

        vp2 = vel[0:3]
        vr2 = vel[3:6]
        dv = q * glm.quat(0, 0, -1, 0) * glm.inverse(q)
        dv = np.array([dv[0], dv[1], dv[2]]) * 0.045 * vel[-1]
        vp1 = vp2 + dv
        dw = q * glm.quat(0,0,0,1)*glm.inverse(q)
        dw = np.array([dw[0], dw[1], dw[2]])*vel[-1]
        vr1 = vr2 + dw
        return pos1, pos2, quat1, quat2, vp1, vp2, vr1, vr2


    def _chop_kin(self, t):
        pose, _ = self.openloop_generator.openloop_full(t)
        self.sim_chopsticks.data.qpos[0:7] = pose[0:7].copy()
        self.sim_chopsticks.forward()

        quat_palm = convert_quat2glm(self.sim_chopsticks.data.body_xquat[self.sim_chopsticks.model.body_name2id('palm')])
        pos_palm = convert_pos2glm(self.sim_chopsticks.data.body_xpos[self.sim_chopsticks.model.body_name2id('palm')])
        quat_chop2 = quat_palm * self.dq_chopstick2
        pos_chop2 = pos_palm + quat_palm * self.dr_chopstick2 * glm.inverse(quat_palm)

        quat_chop1 = quat_chop2 * glm.quat(np.cos(pose[-1]/2), 0, 0, np.sin(pose[-1]/2))
        pos_chop1 = pos_chop2 + quat_chop2 * glm.quat(0,0.045, 0.035, 0) * glm.inverse(quat_chop2) + quat_chop1 * glm.quat(0, -0.045, 0, 0) * glm.inverse(quat_chop1)

        return convert_glm2pos(pos_chop2), convert_glm2quat(quat_chop2), convert_glm2pos(pos_chop1), convert_glm2quat(quat_chop1)


    def _move_reference_chopsticks(self, t):
        pose, _ = self.openloop_generator.chopsticks(t + self.time_step)
        pos = pose[0:3]
        quat = pose[3:7]
        self.sim.data.qpos[self.dof_control + 14 : self.dof_control + 14 + 3] = pos
        self.sim.data.qpos[self.dof_control + 14 + 3 : self.dof_control + 14 + 3 + 4] = quat
        self.sim.data.qpos[self.dof_control + 14 + 3 + 4] = pose.copy()[-1]
        self.sim.forward()


    # Environment methods
    # ----------------------------

    def step(self, action):
        # set the action of the little fingure to zero
        action[30:36] = 0

        #step the simulation
        frames = []
        for substeps in range(self.frame_skip):
            # move the hand
            self.t = self.num_steps * (self.time_step * self.frame_skip) + self.time_step * substeps

            openloop_arm, vel_arm = self.openloop_generator.openloop(self.t)
            base_pose = self.standard_pose[0:self.dof_control].copy() # the rest pose
            base_pose[0:7] = openloop_arm
            target_pos = action + base_pose
            target_vel = np.zeros(self.dof_control)
            target_vel[0:7] = vel_arm

            torque = self._compute_torque(target_pos, target_vel)

            self.sim.data.ctrl[:] = torque
            self.sim.step()

            # move the reference chopsticks
            self._move_reference_chopsticks(self.t + self.time_step)

            # save simulation frames
            frames.append(self.sim.data.qpos[:].copy())

        self.render(self.render_mode)

        self.num_steps += 1
        self.num_steps_sim += 1
        ob = self._get_obs()
        rwd = 0
        self.t = (self.num_steps) * (self.time_step * self.frame_skip)

        idx = self.openloop_generator.object_idx_index(self.t)
        self.object_idx = idx
        self.body_object = self.sim.model.body_name2id("object_" + str(self.object_idx))
        self.geom_object = self.sim.model.geom_name2id("object_" + str(self.object_idx))

        # compute rewards
        rwd_hand = self._compute_rwd_hand()
        rwd_chop1, rwd_chop2 = self._compute_rwd_chopsticks()
        rwd_object = self._compute_rwd_object()
        rwd_contact, rwd_collision = self._compute_rwd_contact_and_collision()
        rwd = rwd_hand * rwd_chop1 * rwd_chop2 * rwd_object * rwd_contact + rwd_collision

        # log infos
        infos = {} 
        infos['frames'] = np.array(frames)
        infos['rwd_chop1'] = rwd_chop1
        infos['rwd_chop2'] = rwd_chop2
        infos['num_steps_sim'] = self.num_steps_sim
        infos['rwd'] = rwd
        infos['pose_rwd'] = rwd_contact

        dr,dq,_,_ = self._get_chopsticks_drdq()
        infos['dr'] = convert_glm2pos(dr)
        infos['dq'] = convert_glm2quat(dq)

        if(self.openloop_generator.phase_index(self.t + self.frame_skip * self.time_step) == 0 and self.openloop_generator.phase_index(self.t) == 3):
            self._reset_objects(self.openloop_generator.object_idx_index(self.t + self.time_step * self.frame_skip))

        done = (rwd_chop1 < 0.5 or rwd_chop2 < 0.5 or rwd_object < 0.7 or rwd_contact < 0.5)
        
        return ob, rwd, done, infos


    def _get_obs(self):
        '''
        Gets the observation of the environment
        '''
        qp_hand = self.sim.data.qpos.ravel().copy()[7:self.dof_control]
        qv_hand = self.sim.data.qvel.ravel().copy()[7:self.dof_control]
        feature_hand = [qp_hand, qv_hand]
    
        pos_chopstick1_sim = self.sim.data.body_xpos[self.body_chopstick1].copy()
        quat_chopstick1_sim = self.sim.data.body_xquat[self.body_chopstick1].copy()
        velp_chopstick1_sim = self.sim.data.body_xvelp[self.body_chopstick1].copy()
        velr_chopstick1_sim = self.sim.data.body_xvelr[self.body_chopstick1].copy()

        pos_chopstick2_sim = self.sim.data.body_xpos[self.body_chopstick2].copy()
        quat_chopstick2_sim = self.sim.data.body_xquat[self.body_chopstick2].copy()
        velp_chopstick2_sim = self.sim.data.body_xvelp[self.body_chopstick2].copy()
        velr_chopstick2_sim = self.sim.data.body_xvelr[self.body_chopstick2].copy()

        t = (self.num_steps+1) * (self.time_step * self.frame_skip)
        pose,vel = self.openloop_generator.chopsticks(t)
        pos1_ref, pos2_ref, quat1_ref, quat2_ref, vp1_ref, vp2_ref, vr1_ref, vr2_ref = self._compute_kinematics(t) 
      
        qp_arm = self.sim.data.qpos.ravel().copy()[0:7]
        qv_arm = self.sim.data.qvel.ravel().copy()[0:7]
        pos_arm_ref, vel_arm_ref = self.openloop_generator.openloop(t)
        pos_arm_ref1, vel_arm_ref1 = self.openloop_generator.openloop(t + 1 * self.frame_skip * self.time_step)
        pos_arm_ref2, vel_arm_ref2 = self.openloop_generator.openloop(t + 2 * self.frame_skip * self.time_step)
        pos_arm_ref3, vel_arm_ref3 = self.openloop_generator.openloop(t + 3 * self.frame_skip * self.time_step)
        pos_arm_ref4, vel_arm_ref4 = self.openloop_generator.openloop(t + 4 * self.frame_skip * self.time_step)
        pos_arm_ref5, vel_arm_ref5 = self.openloop_generator.openloop(t + 5 * self.frame_skip * self.time_step)
      
        dq_arm = qp_arm - pos_arm_ref
        dv_arm = qv_arm - vel_arm_ref
        feature_arm = [qp_arm, qv_arm, pos_arm_ref, vel_arm_ref, pos_arm_ref1, vel_arm_ref1, pos_arm_ref2, vel_arm_ref2,pos_arm_ref3, vel_arm_ref3, pos_arm_ref4, vel_arm_ref4, pos_arm_ref5, vel_arm_ref5, dq_arm, dv_arm]
      
        pose_chopstick_current_ref,_ = self.openloop_generator.chopsticks(t - self.time_step * self.frame_skip)
        pose_object_current_ref, _ = self.openloop_generator.object(t - self.time_step * self.frame_skip)
        contact_mask_ref = self.openloop_generator.contact(t - self.time_step * self.frame_skip)
        quat_chopstick_current_ref = convert_quat2glm(pose_chopstick_current_ref[3:7])
        dr_object_ref = convert_glm2pos( glm.inverse(quat_chopstick_current_ref) * convert_pos2glm(pose_object_current_ref[0:3] - pose_chopstick_current_ref[0:3]) * ( quat_chopstick_current_ref))
        dv_object_ref = np.array([0,0,0])

        geom_type = np.array([self.sim.model.geom_type[self.geom_object]] * 10)
        geom_size = np.concatenate([self.sim.model.geom_size[self.geom_object]] * 10)
       
        pos_object_sim = self.sim.data.body_xpos[self.body_object].copy()
        velp_object_sim = self.sim.data.body_xvelp[self.body_object].copy()
        quat_chopstick_sim = convert_quat2glm(quat_chopstick2_sim)
        dr_object_sim = convert_glm2pos( glm.inverse(quat_chopstick_sim) * convert_pos2glm(pos_object_sim - pos_chopstick2_sim)*( quat_chopstick_sim))
        dv_object_sim = convert_glm2pos( glm.inverse(quat_chopstick_sim) * convert_pos2glm(velp_object_sim - velp_chopstick2_sim)*( quat_chopstick_sim))
        feature_object = [dr_object_ref, dv_object_ref, dr_object_sim, dv_object_sim, geom_type, geom_size]
     
        quat_palm_sim = convert_quat2glm(self.sim.data.body_xquat[self.body_palm])
        pos_palm_sim = self.sim.data.body_xpos[self.body_palm]
        velp_palm_sim = self.sim.data.body_xvelp[self.body_palm]
        velr_palm_sim = self.sim.data.body_xvelr[self.body_palm]

        quat_chopstick2_sim = convert_quat2glm(quat_chopstick2_sim)
        dq_chopstick2_sim = convert_glm2quat( glm.inverse(quat_palm_sim) * quat_chopstick2_sim )
        dr_chopstick2_sim = convert_glm2pos( glm.inverse(quat_palm_sim) * convert_pos2glm(pos_chopstick2_sim - pos_palm_sim) * quat_palm_sim )

        dq_chopstick2_ref = convert_glm2quat(self.dq_chopstick2)
        dr_chopstick2_ref = convert_glm2pos(self.dr_chopstick2)

        diff_dq_chopstick2 = convert_glm2quat( glm.inverse(convert_quat2glm(dq_chopstick2_sim)) * convert_quat2glm(dq_chopstick2_ref))
        diff_dr_chopstick2 = dr_chopstick2_sim - dr_chopstick2_ref

        dvp_chopstick2_ref = np.array([0,0,0])
        dvr_chopstick2_ref = np.array([0,0,0])

        dvp_chopstick2_sim = convert_glm2pos( glm.inverse(quat_palm_sim) * convert_pos2glm(velp_chopstick2_sim - velp_palm_sim) * quat_palm_sim )
        dvr_chopstick2_sim = convert_glm2pos( glm.inverse(quat_palm_sim) * convert_pos2glm(velr_chopstick2_sim - velr_palm_sim) * quat_palm_sim )

        diff_dvp_chopstick2 = dvp_chopstick2_sim - dvp_chopstick2_ref
        diff_dvr_chopstick2 = dvr_chopstick2_sim - dvr_chopstick2_ref
        
        feature_passive = [dq_chopstick2_sim, dr_chopstick2_sim, dq_chopstick2_ref, dr_chopstick2_ref, diff_dq_chopstick2, diff_dr_chopstick2, \
            dvp_chopstick2_sim, dvr_chopstick2_sim, dvp_chopstick2_ref, dvr_chopstick2_ref, diff_dvp_chopstick2, diff_dvr_chopstick2]
       
        quat_chopstick1_sim = convert_quat2glm(quat_chopstick1_sim)
        dq_chopstick1_sim = convert_glm2quat( glm.inverse(quat_palm_sim) * quat_chopstick1_sim )
        dr_chopstick1_sim = convert_glm2pos( glm.inverse(quat_palm_sim) * convert_pos2glm(pos_chopstick1_sim - pos_palm_sim) * quat_palm_sim )

        quat_palm_ref = convert_quat2glm(quat2_ref) * glm.inverse(self.dq_chopstick2)
        pos_palm_ref = pos2_ref - convert_glm2pos( quat_palm_ref * self.dr_chopstick2 * glm.inverse(quat_palm_ref))

        dq_chopstick1_ref = convert_glm2quat( self.dq_chopstick2 * glm.quat(np.cos(pose[-1]/2), 0, 0, np.sin(pose[-1]/2)) )
        dr_chopstick1_ref = convert_glm2pos( glm.inverse(quat_palm_ref) * convert_pos2glm(pos1_ref - pos_palm_ref) * quat_palm_ref )

        diff_dq_chopstick1 = convert_glm2quat( glm.inverse(convert_quat2glm(dq_chopstick1_sim)) * convert_quat2glm(dq_chopstick1_ref))
        diff_dr_chopstick1 = dr_chopstick1_sim - dr_chopstick1_ref

        dvp_chopstick1_ref = convert_glm2pos( glm.inverse(quat_palm_ref) * convert_pos2glm(vp1_ref - vp2_ref) * quat_palm_ref )
        dvr_chopstick1_ref = convert_glm2pos( glm.inverse(quat_palm_ref) * convert_pos2glm(vr1_ref - vr2_ref) * quat_palm_ref )

        dvp_chopstick1_sim = convert_glm2pos( glm.inverse(quat_palm_sim) * convert_pos2glm(velp_chopstick1_sim - velp_palm_sim) * quat_palm_sim )
        dvr_chopstick1_sim = convert_glm2pos( glm.inverse(quat_palm_sim) * convert_pos2glm(velr_chopstick1_sim - velr_palm_sim) * quat_palm_sim )

        diff_dvp_chopstick1 = dvp_chopstick1_sim - dvp_chopstick1_ref
        diff_dvr_chopstick1 = dvr_chopstick1_sim - dvr_chopstick1_ref
        
        feature_active = [dq_chopstick1_sim, dr_chopstick1_sim, dq_chopstick1_ref, dr_chopstick1_ref, diff_dq_chopstick1, diff_dr_chopstick1, \
            dvp_chopstick1_sim, dvr_chopstick1_sim, dvp_chopstick1_ref, dvr_chopstick1_ref, diff_dvp_chopstick1, diff_dvr_chopstick1]
       
        sensored_force = self.sim.data.sensordata[:].copy()
        feature_dynamic = [sensored_force] * 5 + 10 * [np.array([contact_mask_ref])]
        dis_end12object = 1e6
        dis_end22object = 1e6
        dis_contact_thumb, dis_contact_index, dis_contact_middle, dis_contact_ring, dis_contact_little, dis_end12object, dis_end22object = self._compute_dis(t - self.frame_skip * self.time_step)      
        feature_contact_dis = 10 * [np.array([dis_end12object, dis_end22object, dis_contact_thumb, dis_contact_index, dis_contact_middle, dis_contact_ring, dis_contact_little])]
        feature_dynamic += feature_contact_dis
        feature_style = [np.array([self.openloop_generator.rel_pos])] * 10
    
        return np.concatenate(feature_hand + feature_arm + feature_passive + feature_active + feature_dynamic + feature_object + feature_style)


    def _reset_objects(self, object_idx):
        num_objects = self.openloop_generator.object_idx[-1] + 1
        shape = self.openloop_generator.q_init[8:].shape[0]
        self.sim.data.qpos[self.dof_hand_chopstick: self.dof_hand_chopstick + shape] = self.openloop_generator.q_init[8:]
        self.sim.data.qvel[self.dof_hand_chopstick-3:] = 0
        for i in range(num_objects):
           if(i == object_idx):
               continue
           else:
               self.sim.data.qpos[self.dof_hand_chopstick + 7 * i: self.dof_hand_chopstick + 7 * i + 7] = np.array([-10 * (i+1), -10 * i, -10 * i, 1,0,0,0])
        self.sim.forward()
    
    
    def _reset_sim(self):
        self.num_steps = 0
        self.num_steps_nothold = 0

        self.sim.set_state(self.initial_state)

        if(self.save_state == False):
            phase = np.random.randint(0, 4)
            sample_bool = self.sample_pool.len_phase(phase) > 0
            if(np.random.rand()<0.3 and sample_bool):
                state = self.sample_pool.sample(phase)
                if(self.mode == 'batch'):
                    self.openloop_idx = state['openloop_idx']
                    self.openloop_generator = self.openloop_generator_list[self.openloop_idx]
                    self.episode_length = np.around(self.openloop_generator.motion_time/(self.time_step * self.frame_skip))
                
                self.dr_chopstick2 = self.openloop_generator.dr_chopstick2
                self.dq_chopstick2 = self.openloop_generator.dq_chopstick2
                if(type(self.openloop_generator.plate_pos)!= int):
                    self.sim.model.geom_size[-1] = self.openloop_generator.plate_size
                
                num_objects_traj = self.openloop_generator.num_objects
                self.sim.model.geom_size[-num_objects_traj: ] = self.openloop_generator.geom_size.copy()
                
                self.sim.data.qpos[:] = state['qpos'].copy()
                self.sim.data.qvel[:] = state['qvel'].copy()
                self.sim.data.ctrl[:] = state['ctrl'].copy()
                self.dr_rotaxis1, self.dr_rotaxis2 = state['dr_rotaxis1'].copy(), state['dr_rotaxis2'].copy()
                self.sim.data.qacc_warmstart[:] = state['qacc'].copy()
                self.sim.forward()
                self.num_steps = state['num_steps']
                self.num_steps_sim = state['num_steps_sim']             
            else:
                if(self.mode == 'batch'):
                    prob = np.exp(-self.rwd_task/1)
                    prob = prob/prob.sum()
                    self.openloop_idx = find_index(prob, np.random.rand())
                    self.openloop_generator = self.openloop_generator_list[self.openloop_idx]
                    self.episode_length = np.around(self.openloop_generator.motion_time/(self.time_step * self.frame_skip))
                ##########################################
                #change site positions
                self.dr_chopstick2 = self.openloop_generator.dr_chopstick2
                self.dq_chopstick2 = self.openloop_generator.dq_chopstick2
                self._construct_tipdict()
                ##############################################################
                self.num_steps_sim = 0
                if(self.mode =='batch'):
                    start_t = self.openloop_generator.sample_time()
                else:
                    start_t = self.openloop_generator.sample_time(0)
                
                idx = self.openloop_generator.object_idx_index(start_t)
                self.object_idx = idx
                self.body_object = self.sim.model.body_name2id("object_" + str(self.object_idx))
                self.geom_object = self.sim.model.geom_name2id("object_" + str(self.object_idx))
                
                self.num_steps = int(np.round(start_t/(self.frame_skip * self.time_step)))
                pose_arm, vel_arm = self.openloop_generator.openloop(start_t)
                
                num_objects_traj = self.openloop_generator.num_objects
                self.sim.model.geom_size[-num_objects_traj: ] = self.openloop_generator.geom_size.copy()
                
                self.sim.data.qpos[0:7] = pose_arm
                self.sim.data.qvel[0:7] = vel_arm
                self.sim.data.qpos[7:self.dof_control + 0] = self.standard_pose[7: self.dof_control + 0]
                r_chop2, q_chop2, r_chop1, q_chop1 = self._chop_kin(start_t)
                self.sim.data.qpos[self.dof_control: self.dof_control + 3] = r_chop1
                self.sim.data.qpos[self.dof_control + 3: self.dof_control + 7] = q_chop1
                self.sim.data.qpos[self.dof_control + 7: self.dof_control + 10] = r_chop2
                self.sim.data.qpos[self.dof_control + 10: self.dof_control + 14] = q_chop2

                self._reset_objects(self.openloop_generator.object_idx_index(start_t))
                if(type(self.openloop_generator.plate_pos)!= int):
                    self.sim.model.geom_size[-1] = self.openloop_generator.plate_size
                num_objects = self.openloop_generator.object_idx[-1] + 1
                if(type(self.openloop_generator.plate_pos)!= int):
                    self.sim.data.qpos[self.dof_hand_chopstick + 7 * num_objects: self.dof_hand_chopstick + 7 * num_objects + 3] = self.openloop_generator.plate_pos
                self.sim.data.ctrl[:] = 0#self.init_ctrl.copy()
                self.sim.data.qacc_warmstart[:] = 0
                self.sim.forward()
        else:
            self.num_steps_sim = 0
            if(self.mode == 'batch'):
                self.openloop_generator = self.openloop_generator_list[self.openloop_idx]
                self.episode_length = np.around(self.openloop_generator.motion_time/(self.time_step * self.frame_skip))

            self.dr_chopstick2 = self.openloop_generator.dr_chopstick2
            self.dq_chopstick2 = self.openloop_generator.dq_chopstick2
            self._construct_tipdict()
            
            if(self.mode =='batch'):
                start_t = self.openloop_generator.sample_time()
            else:
                start_t = self.openloop_generator.sample_time(0)
            idx = self.openloop_generator.object_idx_index(start_t)
            self.object_idx = idx
            self.body_object = self.sim.model.body_name2id("object_" + str(self.object_idx))
            self.geom_object = self.sim.model.geom_name2id("object_" + str(self.object_idx))

            self.num_steps = int(np.round(start_t/(self.frame_skip * self.time_step)))
            pose_arm, vel_arm = self.openloop_generator.openloop(start_t)

            num_objects_traj = self.openloop_generator.num_objects
            self.sim.model.geom_size[-num_objects_traj: ] = self.openloop_generator.geom_size.copy()

            self.sim.data.qpos[0:7] = pose_arm
            self.sim.data.qvel[0:7] = vel_arm
            self.sim.data.qpos[7:self.dof_control + 0] = self.standard_pose[7: self.dof_control + 0]
            r_chop2, q_chop2, r_chop1, q_chop1 = self._chop_kin(start_t)
            self.sim.data.qpos[self.dof_control: self.dof_control + 3] = r_chop1
            self.sim.data.qpos[self.dof_control + 3: self.dof_control + 7] = q_chop1
            self.sim.data.qpos[self.dof_control + 7: self.dof_control + 10] = r_chop2
            self.sim.data.qpos[self.dof_control + 10: self.dof_control + 14] = q_chop2
            self._reset_objects(self.openloop_generator.object_idx_index(start_t))
            if(type(self.openloop_generator.plate_pos)!= int):
                self.sim.model.geom_size[-1] = self.openloop_generator.plate_size
            num_objects = self.openloop_generator.object_idx[-1] + 1
            if(type(self.openloop_generator.plate_pos)!= int):
                self.sim.data.qpos[self.dof_hand_chopstick + 7 * num_objects: self.dof_hand_chopstick + 7 * num_objects + 3] = self.openloop_generator.plate_pos
            self.sim.data.ctrl[:] = 0
            self.sim.data.qacc_warmstart[:] = 0
            self.sim.forward()
        
        self.sim.forward()
        return self._get_obs()
