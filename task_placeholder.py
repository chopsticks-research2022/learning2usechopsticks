import copy
import json
import os

import glm
import mujoco_py
import numpy as np
import xmltodict
from IPython import embed
from scipy.spatial.transform import Rotation as R

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from algorithm.human_arm_ik_solver import HumanArm_IK_Solver
from envs.motiongenerator import OpenloopGenerator
from model.grasping_model import GraspingModel
from model.trajectory_planner import TrajectoryPlanner
from utils.convert_util import *


def compute_anglevel(q0, q1, dt):
    #compute the angular velocity of the object in the global coordinate system
    #q0: inital quaternion
    #q1: end quaternion
    #dt : time interval
    dq = q1 * glm.inverse(q0)
    angle = np.arccos(np.clip(dq[3], -1, 1)) * 2
    if(angle <= 1e-6):
        w = np.array([0, 0, 0])
    else:
        axis = np.array([dq[0]/np.sin(angle/2), dq[1]/np.sin(angle/2), dq[2]/np.sin(angle/2)])
        w = axis * angle / dt
    return w

def convertangle2quat(theta, axis = 0):
    if(axis == 0):
        return glm.quat(np.cos(theta/2), np.sin(theta/2), 0, 0)
    elif(axis == 1):
        return glm.quat(np.cos(theta/2), 0, np.sin(theta/2), 0)
    else:
        return glm.quat(np.cos(theta/2), 0, 0, np.sin(theta/2))

def compute_vel(motion, type = 'motion'):
    '''
    given the motion of the chopsticks or object, compute the velocities
    '''
    if(type == 'motion'):
        vel = np.zeros((motion.shape[0], motion.shape[1] - 1))
        if(motion.shape[1] == 9):
            for i in range(vel.shape[0]-1):
                vel[i, 1: 1+ 3] = (motion[i+1, 1:1+3] - motion[i, 1:1+3])/motion[0,0]
                vel[i, 1+3: 1 + 3 + 3] = compute_anglevel(convert_quat2glm(motion[i, 1+3: 1 +3 + 4]), convert_quat2glm(motion[i+1, 1+3 : 1+3+4]), motion[0,0])
                vel[i,-1] = (motion[i+1, -1] - motion[i,-1])/motion[0,0]
            vel[-1,1:] = vel[-2,1:].copy()
            pass
        elif(motion.shape[1] == 8):
            for i in range(vel.shape[0]-1):
                vel[i, 1: 1+ 3] = (motion[i+1, 1:1+3] - motion[i, 1:1+3])/motion[0,0]
                vel[i, 1+3:] = compute_anglevel(convert_quat2glm(motion[i, 1+3:]), convert_quat2glm(motion[i+1, 1+3 :]), motion[0,0])
            vel[-1,1:] = vel[-2,1:].copy()
        else:
            raise NotImplementedError
    elif(type == 'openloop'):
        vel = np.zeros((motion.shape[0], motion.shape[1]))
        vel[:,0] = motion[:,0]
        vel[0:-1,1:] = (motion[1:,1:] - motion[0:-1,1:])/motion[0,0]
        vel[-1,1:] = vel[-2,1:].copy()
    else:
        raise NotImplementedError
    return vel

def grasp_task_holder(chop_xml, pose_file, transform_chopsticks_start, objects_paras, transform_objects_start, transform_objects_end, thumb_relative_position = 0, save_folder= None, task_path = None, plate_sampler = None):
    '''
    given a task description, we generate a sequences of planning motions
    '''
    grasp_nn = GraspingModel('./data/grasp/graspnet_mlp.pth', './data/reach/reachnet_mlp.pth')
    def save_files(data, save_path):
        '''
        save the jason file
        '''
        for key in data.keys():
            if(type(data[key]) == str):
                pass
            elif(type(data[key]) == float):
                pass
            elif(type(data[key]) == int):
                pass
            else:
                data[key] = data[key].tolist()
        with open(save_path, 'w') as outfile:
            json.dump(data, outfile)

    def save_as_dict(motions_chop, motions_arm, motions_object, xml_string_kin, xml_string_sim, object_idx = 0, contact = True, phase = 0):
        data = {}
        data['openloop_motion'] = np.concatenate([motions_arm, motions_chop[:,-1].reshape((-1,1))],1)
        data['vel_openloop'] = compute_vel(np.concatenate([motions_arm, motions_chop[:,-1].reshape((-1,1))],1), 'openloop')
        data['openloop_arm'] = motions_arm
        data['vel_openloop_arm'] = compute_vel(motions_arm, 'openloop')
        data['motion_chopsticks'] = motions_chop
        data['vel_chopsticks'] = compute_vel(motions_chop, 'motion')
        data['motion_object'] = motions_object
        data['vel_object'] = compute_vel(motions_object, 'motion')
        data['contact'] = np.array([contact] * motions_arm.shape[0])
        data['phase'] = np.array([phase] * motions_arm.shape[0])
        data['xml_kin'] = xml_string_kin
        data['xml_sim'] = xml_string_sim
        data['object_idx'] = np.array([object_idx] * motions_arm.shape[0])
        return data


    def computedrdq(transform_object, transform_chopsticks):
        #compute the relative position and rotation of the object
        quat_chopstick2 = convert_quat2glm(transform_chopsticks[3:7])
        quat_object = convert_quat2glm(transform_object[3:])
        dq = glm.inverse(quat_chopstick2) * quat_object
        dr = transform_object[0:3] - transform_chopsticks[0:3]
        dr = glm.quat(0, dr[0], dr[1], dr[2])
        dr = glm.inverse(quat_chopstick2) * dr * quat_chopstick2
        return dr, dq

    def compute_transform_chopsticks_from_object(transform_object, dr, dq, theta):
        '''
        given tranformation of the object, relative translation and rotation , compute the transformation of the chopsticks
        '''
        quat_chopsticks = convert_quat2glm(transform_object[3:7]) * glm.inverse(dq)
        pos_chopsticks = transform_object[0:3] - convert_glm2pos(quat_chopsticks * dr * glm.inverse(quat_chopsticks))
        return np.concatenate([pos_chopsticks, convert_glm2quat(quat_chopsticks), np.array([theta])]) 

    def compute_transform_object_from_chopsticks(transform_chopsticks, dr, dq):
        '''
        given transform of the chopsticks, dr and dq, compute the transformation of the object
        '''
        q = convert_glm2quat(convert_quat2glm(transform_chopsticks[3:7]) * dq)
        pos = transform_chopsticks[0:3] + convert_glm2pos(convert_quat2glm(transform_chopsticks[3:7]) * dr * glm.inverse(convert_quat2glm(transform_chopsticks[3:7])))
        return np.concatenate([pos,q]) 

    def convert_chopsticks2object(motions_chopsticks, dr, dq):
        '''
        transform the chopsticks motions into object motions
        '''
        motions_object = np.zeros((motions_chopsticks.shape[0], 1+ 7))
        motions_object[:,0] = 0.01
        for i in range(motions_object.shape[0]):
            motions_object[i,1:] = compute_transform_object_from_chopsticks(motions_chopsticks[i, 1:], dr, dq)
        return motions_object

    def release_motion(transform_chopsticks, task = 'moving'):
        '''
        release the chopsticks to let the object slip, for 2 tasks: grasp and throw
        for throwing tasks, directly release the chopsticks
        for grasping tasks, stay static for a few moments and then release the chopsticks
        '''
        if(task == 'moving' or task == 'catching'):
            t = 0.05
            motions_chopsticks_static = np.zeros((5, 1 + 3 + 4 + 1))
            motions_chopsticks_static[:,0] = 0.01
            motions_chopsticks_static[:,1:] = transform_chopsticks

            motions_release_chopsticks = np.zeros((10, 1 + 8))
            motions_release_chopsticks[:,0] = 0.01
            for i in range(10):
                motions_release_chopsticks[i, 1:] = transform_chopsticks.copy()
                motions_release_chopsticks[i, -1] = transform_chopsticks[-1] + (0 - transform_chopsticks[-1]) * i/9
            motions_reset_chopsticks = np.zeros((11, 1 + 8))
            motions_reset_chopsticks[:-1,0] = 0.01
            for i in range(11):
                theta  = motions_release_chopsticks[-1,-1] + (0 - motions_release_chopsticks[-1,-1]) * i / 10
                motions_reset_chopsticks[i, 1:] = transform_chopsticks.copy()
                motions_reset_chopsticks[i, 3] += 0.001 * i
                motions_reset_chopsticks[i,-1] = theta
            return np.concatenate([motions_chopsticks_static, motions_release_chopsticks, motions_reset_chopsticks])

        elif(task == 'throwing'):
            motions_release_chopsticks = np.zeros((10, 1 + 8))
            motions_release_chopsticks[:,0] = 0.01
            for i in range(10):
                motions_release_chopsticks[i,1:] = transform_chopsticks - np.array([0,0,0,0,0,0,0,0.005]) * i
            motions_reset_chopsticks = np.zeros((11, 1 + 8))
            motions_reset_chopsticks[:-1, 0] = 0.01
            for i in range(11):
                theta  = motions_release_chopsticks[-1,-1] + (0 - motions_release_chopsticks[-1,-1]) * i / 10
                motions_reset_chopsticks[i,1:] = transform_chopsticks.copy()
                motions_reset_chopsticks[i,-1] = theta
            return np.concatenate([motions_release_chopsticks, motions_reset_chopsticks])
        else:
            raise NotImplementedError

    def lift_motion(transform_chopsticks, task = 'moving'):
        '''
        given the transformation of the chopsticks, generate grasping motions with which to grasp the object
        '''
        if(task == 'moving'  or task == 'throwing'):
            #first stay static for a few moments
            motions_chopsticks_static = np.zeros((5, 1 + 3 + 4 + 1))
            motions_chopsticks_static[:,0] = 0.01
            motions_chopsticks_static[:,1:] = transform_chopsticks
            #grasp the object
            motions_chopsticks_grasp = np.zeros((10, 1 + 3 + 4 + 1))
            motions_chopsticks_grasp[:,0] = 0.01
            motions_chopsticks_grasp[:,1:] = transform_chopsticks
            for i in range(10):
                motions_chopsticks_grasp[i,-1] =  transform_chopsticks[-1] + 0.08 #+ (0.2 - transform_chopsticks[-1]) * i /9
            #lift the object
            motions_chopsticks_lift = np.zeros((11, 1 + 3 + 4 + 1))
            motions_chopsticks_lift[:-1,0] = 0.01
            motions_chopsticks_lift[:,1:] = motions_chopsticks_grasp[-1,1:]
            for i in range(11):
                motions_chopsticks_lift[i,3] += 0.002 * i
            return np.concatenate([motions_chopsticks_static, motions_chopsticks_grasp, motions_chopsticks_lift])

        elif(task == 'catching'):
            pass
        else:
            raise NotImplementedError

    def cat_data(seq_data, xml_string_kin, xml_string_sim, dr_chopstick2, dq_chopstick2):
        '''
        concatenate the data
        seq_data: sequences of the motion data
        xml_string: xml file for simulation
        aobject_idx: the idx of the object
        '''
        data={}
        openloop_motion = []
        vel_openloop =[]
        openloop_arm = []
        vel_openloop_arm =[]
        motion_object = []
        vel_object = []
        motion_chopsticks=[]
        vel_chopsticks = []
        contacts = []
        phases = [np.array([0])]
        object_idxes = []
        for i in range(len(seq_data)):
            if(i!=len(seq_data) - 1):
                openloop_motion.append(seq_data[i]['openloop_motion'][:-1,:])
                vel_openloop.append(seq_data[i]['vel_openloop'][:-1,:])
                motion_object.append(seq_data[i]['motion_object'][:-1,:])
                vel_object.append(seq_data[i]['vel_object'][:-1,:])
                vel_chopsticks.append(seq_data[i]['vel_chopsticks'][:-1,:])
                motion_chopsticks.append(seq_data[i]['motion_chopsticks'][:-1,:])
                contacts.append(seq_data[i]['contact'][:-1])
                phases.append(seq_data[i]['phase'][1:])
                object_idxes.append(seq_data[i]['object_idx'][:-1])
                openloop_arm.append(seq_data[i]['openloop_arm'][:-1,:])
                vel_openloop_arm.append(seq_data[i]['vel_openloop_arm'][:-1,:])
            else:
                openloop_motion.append(seq_data[i]['openloop_motion'])
                vel_openloop.append(seq_data[i]['vel_openloop'])
                motion_object.append(seq_data[i]['motion_object'])
                vel_object.append(seq_data[i]['vel_object'])
                vel_chopsticks.append(seq_data[i]['vel_chopsticks'])
                motion_chopsticks.append(seq_data[i]['motion_chopsticks'])
                contacts.append(seq_data[i]['contact'])
                phases.append(seq_data[i]['phase'][1:])
                object_idxes.append(seq_data[i]['object_idx'])
                openloop_arm.append(seq_data[i]['openloop_arm'])
                vel_openloop_arm.append(seq_data[i]['vel_openloop_arm'])
        data['openloop_motion'] = np.concatenate(openloop_motion)
        data['vel_openloop'] = np.concatenate(vel_openloop)
        data['q_init'] = seq_data[0]['q_init']
        data['openloop_arm'] = np.concatenate(openloop_arm)
        data['vel_openloop_arm'] = np.concatenate(vel_openloop_arm)
        data['motion_object'] = np.concatenate(motion_object)
        data['motion_chopsticks'] = np.concatenate(motion_chopsticks)
        data['vel_chopsticks'] = np.concatenate(vel_chopsticks)
        data['vel_object'] = np.concatenate(vel_object)
        data['contact'] = np.concatenate(contacts)
        data['phase'] = np.concatenate(phases)
        data['xml_kin'] = xml_string_kin
        data['xml_sim'] = xml_string_sim
        data['object_idx'] = np.concatenate(object_idxes)
        data['dr_chopstick2'] = dr_chopstick2
        data['dq_chopstick2'] = dq_chopstick2
        return data

    
    def generate_xml(xml_path, objects_paras, transform_objects_start, thumb_relative_position = 0, with_arm = False, plate_sampler = None):
        '''
        generate a xml file to locate objects with given initial transformations
        '''
        with open(xml_path, 'r') as fd:
            xml_string = fd.read()
        xml_dict = xmltodict.parse(xml_string)
        object_dict_template = xml_dict['mujoco']['worldbody']['body'][-1]

        #add relative thumb relative position
        body = xml_dict['mujoco']['worldbody']['body'][0]
        while(body['@name'] != 'chopstick2'):
            body = body['body']
        if(with_arm == True):
            body['geom'][-1]['@pos'] = "{} 0 0 ".format(0.045  - thumb_relative_position)
        else:
            body['geom'][-1]['@pos'] = "{} 0 0 ".format(0.045  - thumb_relative_position)
        if('@pos' in body.keys() and '@quat' in body.keys()):
            pos_str = body['@pos'].split()
            quat_str = body['@quat'].split()
            pos_chop2 = np.array([float(pos_str[0]), float(pos_str[1]), float(pos_str[2])])
            quat_chop2 = np.array([float(quat_str[0]), float(quat_str[1]), float(quat_str[2]), float(quat_str[3])])
            quat_chop2 = R.from_quat([quat_chop2[1],quat_chop2[2],quat_chop2[3],quat_chop2[0]])
            relative_pos = quat_chop2.apply(np.array([thumb_relative_position, 0, 0]))
            pos_chop2 = pos_chop2 + relative_pos
            body['@pos'] = '{} {} {}'.format(pos_chop2[0], pos_chop2[1], pos_chop2[2])
        else:
            print('kinematic xml files')

        for i in range(len(objects_paras)-1):
            xml_dict['mujoco']['worldbody']['body'].append(copy.deepcopy(object_dict_template))

        if(plate_sampler!= None):
            for i in range(plate_sampler.num_plates):
                xml_dict['mujoco']['worldbody']['body'].append(copy.deepcopy(object_dict_template))
            
        for i in range(len(objects_paras)):
            dict_object = xml_dict['mujoco']['worldbody']['body'][1 + i] ########here we use 2 since the bowl is also a object
            dict_object['@name'] = 'object_' + str(i)
            dict_object['joint']['@name'] = 'object_' + str(i)
            dict_object['geom']['@name'] = 'object_' + str(i)
            dict_object['@quat'] = '{} {} {} {}'.format(transform_objects_start[i, 3], transform_objects_start[i, 4], transform_objects_start[i, 5], transform_objects_start[i, 6])
            dict_object['geom']['@type'] = objects_paras[i]['type']
            if(objects_paras[i]['type'] == 'sphere'):
                dict_object['geom']['@size'] = '{}'.format(objects_paras[i]['size'][0])
                dict_object['@pos'] = "{} {} {}".format(transform_objects_start[i, 0], transform_objects_start[i, 1], np.max([transform_objects_start[i, 2], objects_paras[i]['size'][0]]))
            elif(objects_paras[i]['type'] == 'box'):
                dict_object['geom']['@size'] = '{} {} {}'.format(objects_paras[i]['size'][0], objects_paras[i]['size'][1], objects_paras[i]['size'][2])
                dict_object['@pos'] = "{} {} {}".format(transform_objects_start[i, 0], transform_objects_start[i, 1], np.max([transform_objects_start[i, 2], objects_paras[i]['size'][2]]))
            elif(objects_paras[i]['type'] == 'capsule'):
                dict_object['geom']['@size'] = '{} {}'.format(objects_paras[i]['size'][0], objects_paras[i]['size'][1])
                dict_object['@pos'] = "{} {} {}".format(transform_objects_start[i, 0], transform_objects_start[i, 1], np.max([transform_objects_start[i, 2], objects_paras[i]['size'][0]]))
                dict_object['geom']['@quat'] = "{} {} {} {}".format(0.707, 0,0.707, 0 )
            else:
                raise NotImplementedError
        if(plate_sampler!=None):
            for i in range(plate_sampler.num_plates):
                ########################################################################
                dict_object = xml_dict['mujoco']['worldbody']['body'][1 + len(objects_paras) + i] ########here we use 2 since the bowl is also a object
                #########################################################################
                dict_object['@name'] = 'plate_' + str(i)
                dict_object['joint']['@name'] = 'plate_' + str(i)
                dict_object['geom']['@name'] = 'plate_' + str(i)
                dict_object['@quat'] = '1 0 0 0'
                dict_object['geom']['@type'] = 'box'
            
                pos = plate_sampler.pos_list[i]
                size = plate_sampler.size_list[i]
                dict_object['geom']['@size'] = '{} {} {}'.format(size[0], size[1], size[2])
                dict_object['geom']['@density'] = '100000'
                dict_object['@pos'] = "{} {} {}".format(pos[0], pos[1], size[2])

        if(with_arm == True):
            xml_string_sim = xmltodict.unparse(xml_dict, pretty=True)
            quat_chop2 = quat_chop2.as_quat()
            return xml_string_sim, pos_chop2, np.array([quat_chop2[3], quat_chop2[0], quat_chop2[1], quat_chop2[2]])
        else:
            xml_string_sim = xmltodict.unparse(xml_dict, pretty=True)
            return xml_string_sim
    def check_continous(a,b):
        if(np.linalg.norm(a[-1,1:] - b[0,1:])>0.1):
            return True

    def move_objects_away(q_init,idx, num_objects):
        q = q_init.copy()
        for i in range(num_objects):
            if(i == idx):
                continue
            else:
                q[8 + i * 7:8 + i *7 + 3] = np.random.uniform([-0.1,-0.5, -0.5], [0.1, -0.5, -0.1])
                q[8 + i *7 + 3: 8 + i *7 + 7] = np.array([1,0,0,0])
        return q
    #start planning
    print('start planning')
    xml_string_sim, dr_chop2, dq_chop2 = generate_xml(chop_xml, objects_paras, transform_objects_start, thumb_relative_position, True, plate_sampler)
    xml_string_kin = generate_xml('./data/chopsticks_xml/chopsticks_template_kin.xml', objects_paras, transform_objects_start, 0, False, plate_sampler)
    model = mujoco_py.load_model_from_xml(xml_string_sim)
    sim = mujoco_py.MjSim(model)
    ik =  HumanArm_IK_Solver(xml_string_sim)
    if(plate_sampler == None):
        num_plates = 0
    else:
        num_plates = plate_sampler.num_plates
    planner = TrajectoryPlanner(xml_string_kin, 8, 8 + transform_objects_start.shape[0] * 7 + 7 * num_plates)
    q_init_arm = ik.ik(transform_chopsticks_start[:-1])
    q_init = sim.data.qpos[:].copy()
    q_init[0:7] = q_init_arm
    q_end = q_init.copy()

    print("enumerate all the objects")
    seq_data_list = []
    dr_list = []
    dq_list = []
    for i in range(transform_objects_start.shape[0]):
        print('processing the: {} object'.format(i))
        q_init = q_end.copy()

        object_para = objects_paras[i]
        x = np.array([[1,0,0,0,0.01,0.01,0.01]])
        if(object_para['type'] == 'sphere'):
            x[0,0:4] = np.array([1,0,0,0])
            x[0,4:7] = object_para['size']
        elif(object_para['type'] == 'box'):
            x[0,0:4] = np.array([0,1,0,0])
            x[0,4:7] = object_para['size']
        elif(object_para['type'] == 'capsule'):
            x[0,0:4] = np.array([0,0,1,0])
            x[0,4:6] = object_para['size']
        else:
            raise NotImplementedError
        print('x:{}'.format(x))
        if (i!= 0):
            transform_chopsticks_start = motions_release_chop[-1,1:].copy()
        import time
        start = time.time()
        transform_chopsticks_pregrasp = grasp_nn.inference(x, transform_objects_start[i].reshape((1,-1)), transform_chopsticks_start[:-1].reshape((1,-1)))
        end = time.time()
        print("grasp inference time:{}".format(end - start))
        dr, dq = computedrdq(transform_objects_start[i], transform_chopsticks_pregrasp)
        dr_list.append(dr)
        dq_list.append(dq)

        q_temp = move_objects_away(q_init, i, transform_objects_start.shape[0])
        motions_pregrasp_chop  = planner.opt_move_seq(q_temp[8:], transform_chopsticks_start, transform_chopsticks_pregrasp, i, dr ,dq,  mode = 'pregrasp')
        print('pregrasp')
        if(type(motions_pregrasp_chop) == bool):
            print('4')
            return False
        motions_pregrasp_arm = ik.ik_seq(motions_pregrasp_chop, q_init[0:7])
        if(type(motions_pregrasp_arm) == bool):
            print('5')
            return False
     
        motions_lift_chop = lift_motion(transform_chopsticks_pregrasp, 'moving')
        motions_lift_arm = ik.ik_seq(motions_lift_chop, motions_pregrasp_arm[-1,1:])
        if(type(motions_lift_arm) == bool):
            print('1')
            return False
        if(check_continous(motions_pregrasp_arm, motions_lift_arm)):
            print('2')
            embed()
      
        #transform_objects_end[i] = ik.solve_pos(i, transform_objects_end[i][0:3], dr, dq, motions_lift_arm[-1, 1:])
        transform_chopsticks_move = compute_transform_chopsticks_from_object(transform_objects_end[i] + np.array([0,0,0.01,0,0,0,0]), dr, dq, transform_chopsticks_pregrasp[-1] + 0.08)
        q_temp = move_objects_away(q_init, i, transform_objects_start.shape[0])
        motions_move_chop = planner.opt_move_seq(q_temp[8:], transform_chopsticks_pregrasp + np.array([0,0,0.02,0,0,0,0,0.08]), transform_chopsticks_move, i, dr, dq, mode = 'moving')
        if(type(motions_move_chop) == bool):
            print('3')
            return False
        motions_move_arm = ik.ik_seq(motions_move_chop, motions_lift_arm[-1,1:])
        if(type(motions_move_arm) == bool):
            return False
        if(check_continous(motions_lift_arm, motions_move_arm)):
            embed()
        print('move')
       
        motions_release_chop = release_motion(transform_chopsticks_move, 'moving')
        motions_release_arm = ik.ik_seq(motions_release_chop, motions_move_arm[-1,1:])
        if(type(motions_release_arm) == bool):
            return False

        motions_pregrasp_object = np.zeros((motions_pregrasp_arm.shape[0], 1 + 7))
        motions_pregrasp_object[:-1,0] = 0.01
        motions_pregrasp_object[:,1:] = np.concatenate([transform_objects_start[i].reshape((1,-1))] * motions_pregrasp_chop.shape[0])

        motions_lift_object = convert_chopsticks2object(motions_lift_chop, dr, dq)
        motions_move_object = convert_chopsticks2object(motions_move_chop, dr, dq)
        
        motions_release_object = np.zeros((motions_release_arm.shape[0], 1 + 7))
        motions_release_object[:-1,0] = 0.01
        motions_release_object[:,1:] = np.concatenate([transform_objects_end[i].reshape((1,-1))] * motions_release_chop.shape[0])


        data_pregrasp = save_as_dict(motions_pregrasp_chop, motions_pregrasp_arm, motions_pregrasp_object, xml_string_kin, xml_string_sim, i, 0, 0)
        data_pregrasp['q_init'] = q_init.copy()
        data_lift = save_as_dict(motions_lift_chop, motions_lift_arm, motions_lift_object, xml_string_kin, xml_string_sim, i, 1, 1)
        data_move = save_as_dict(motions_move_chop, motions_move_arm, motions_move_object, xml_string_kin, xml_string_sim, i, 1, 2)
        data_release = save_as_dict(motions_release_chop, motions_release_arm, motions_release_object, xml_string_kin, xml_string_sim, i, 0, 3)
        seq_data = cat_data([data_pregrasp, data_lift, data_move, data_release], xml_string_kin, xml_string_sim, dr_chop2, dq_chop2)
        q_end = q_init.copy()
        q_end[8 + i * 7:8 + i *7 + 7] = transform_objects_end[i]#hack!!!!!!!!!!!transfrom_objects_end[i]
        q_end[0:7] = motions_release_arm[-1,1:8]
        #print('q_end:{}'.format(q_end))
        seq_data['q_init'] = q_init.copy()
        seq_data['q_end'] = q_end.copy()
        seq_data_list.append(seq_data)
        print(" ")
   
    seq_data_list = cat_data(seq_data_list, xml_string_kin, xml_string_sim, dr_chop2, dq_chop2)
    seq_data_list['rel_pos'] = -thumb_relative_position
   
    with open(pose_file, 'r') as f:
                pose_dict = json.load(f)
    seq_data_list['qpos'] = np.array(pose_dict['qpos'])
    seq_data_list['grasp_mode'] = np.array([1,1,2,0])#np.array(pose_dict['grasp_mode'])
    seq_data_list['tip_pos'] = np.array(pose_dict['tip_pos'])

    save_files(seq_data_list, './data/'+save_folder +'/'+task_path+'.txt')
    print('saving'+ './data/'+ save_folder + '/'+task_path+'.txt')
    return seq_data_list
            

     
    
def render(path):
    with open(path, 'r') as f:
        data = json.load(f)
        xml = data['xml_sim']
        for key in data.keys():
            if(type(data[key]) == list):
                data[key] = np.array(data[key])
    model = mujoco_py.load_model_from_xml(xml)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    mog2 = OpenloopGenerator(data)
    t = 0
    q_seq = []
    #sim.data.qpos[:] = data['q_init']
    sim.forward()
    q_init = sim.data.qpos[:].copy()
    data = []
    while(1):
        if(t> mog2.motion_time):
           t= 0
           sim.data.qpos[:] = q_init.copy()
        pose_sim, _, = mog2.openloop_full(t)
        pose_object, _ = mog2.object(t)
        idx = mog2.object_idx_index(t)
        data.append(pose_sim)
        sim.data.qpos[0:8] = pose_sim
        sim.data.qpos[8 + idx * 7: 8 + idx * 7 + 7] = pose_object
        sim.forward()
        viewer.render()
        q_seq.append(sim.data.qpos[:].copy())
        t+=0.01
        t = round(t,5)


class PlateSampler():
    '''
    this class is used to  sample various plates(heights variation)
    '''
    def __init__(self, num_plates = 0, bounds = np.array([[-0.3, -0.],[-0.1, 0.1]])):
        self.num_plates = num_plates
        self.pos_list = []
        self.size_list = []
        num_plates = 0
        while(num_plates< self.num_plates):
            pos = np.random.uniform(bounds[0], bounds[1])
            size = np.random.uniform(np.array([0.03,0.03,0.03]), np.array([0.05, 0.05, 0.1]))
            if(num_plates == 0):
                self.pos_list.append(pos)
                self.size_list.append(size)
                num_plates += 1
            else:
                flag = False
                while(not flag):
                    pos_past = np.array(self.pos_list)[:,0:2] # (N, 2)
                    size_past = np.array(self.size_list) #(N, 3)
                    pos = np.random.uniform(bounds[0], bounds[1])
                    size = np.random.uniform(np.array([0.03,0.03,0.05]), np.array([0.05, 0.05, 0.1]))
                    x_dist = np.abs(pos_past[:, 0] - pos[0])
                    y_dist = np.abs(pos_past[:, 1] - pos[1])
                    collide =  np.logical_and((x_dist < size_past[:,0] + size[0]), ( y_dist < size_past[:, 1] + size[1]))
                    if(collide.sum() > 0):
                        continue
                    else:
                        flag = True
                        self.pos_list.append(pos)
                        self.size_list.append(size)
                        num_plates += 1

    def read_plate_info(self, idx):
        return self.pos_list[idx], self.size_list[idx]

class PositionSampler():
    '''
    this class is used to sample a list of objects
    '''
    def __init__(self, num_objects = 6, bounds_start = np.array([[-0.075, -0.075],[0.075, 0.0]]), bounds_end = np.array([[-0.075, -0.05],[0.075, 0.0]])):
        self.num_objects = num_objects
        self.start_position = []
        self.end_position = []
        self.bounds_start = bounds_start
        self.bounds_end = bounds_end
        
    def sample(self):
        num = 0
        while(num < self.num_objects + 1):
            pos = np.random.uniform(self.bounds_start[0], self.bounds_start[1])
            pos = np.array([pos[0], pos[1], 0])
        
            self.start_position.append(pos.copy())
            pos = np.random.uniform(self.bounds_end[0], self.bounds_end[1])
            pos = np.array([pos[0], pos[1], 0])
            
            self.end_position.append(pos.copy())
            num += 1

    def delete(self):
        self.start_position = []
        self.end_position = []
        
def readdrdq_fromxml(xml):
    xml_dict = xmltodict.parse(xml)
    body = xml_dict['mujoco']['worldbody']['body'][0]
    while(body['@name'] != 'chopstick2'):
        body = body['body']
    return body['@pos'], body['@quat']
      
def generate_xml(num_objects, chop_xml_path, hand_xml_path, xml_name = None, objects_paras = None, plate_sampler = None):
    '''
    generate hand simulation xml files
    input:
        num_objects: the number of objects
        chop_xml_path: the xml file used for motion planning
        hand_xml_path: the hand simulation template xml files
        xml_name: the name of our final xml files
        objects_paras: the parameters of all objects
        plate_sampler: add heights variations
    '''
    #read dr and dq from the chopsticks xml file
    with open(chop_xml_path, 'r') as fd:
        xml_string = fd.read()
    dr, dq = readdrdq_fromxml(xml_string)

    #modify the xml files for training 
    with open(hand_xml_path, 'r') as fd:
        xml_hand_string = fd.read()
    xml_hand_dict = xmltodict.parse(xml_hand_string)
    object_dict_template = xml_hand_dict['mujoco']['worldbody']['body'][-1]
    #save contact parameters
    for i in range(num_objects):
        contact_sample = copy.deepcopy(xml_hand_dict['mujoco']['contact']['pair'][0])
        contact_sample['@geom1'] = 'chopstick1_end'
        contact_sample['@geom2'] = 'object_' + str(i)
        contact_sample['@margin'] = '10'
        contact_sample['@gap'] = '100'
        xml_hand_dict['mujoco']['contact']['pair'].append(contact_sample)

        contact_sample = copy.deepcopy(xml_hand_dict['mujoco']['contact']['pair'][0])
        contact_sample['@geom1'] = 'chopstick2_end'
        contact_sample['@geom2'] = 'object_' + str(i)
        contact_sample['@margin'] = '10'
        contact_sample['@gap'] = '100'
        xml_hand_dict['mujoco']['contact']['pair'].append(contact_sample)

        contact_sample = copy.deepcopy(xml_hand_dict['mujoco']['contact']['pair'][0])
        contact_sample['@geom1'] = 'chopstick1'
        contact_sample['@geom2'] = 'object_' + str(i)
        contact_sample['@margin'] = '0'
        contact_sample['@gap'] = '0'
        contact_sample['@condim'] = '4'
        contact_sample['@solimp'] = "0.99 0.99 0.01"
        contact_sample['@solref'] = "0.01 1"
        xml_hand_dict['mujoco']['contact']['pair'].append(contact_sample)

        contact_sample = copy.deepcopy(xml_hand_dict['mujoco']['contact']['pair'][0])
        contact_sample['@geom1'] = 'chopstick2'
        contact_sample['@geom2'] = 'object_' + str(i)
        contact_sample['@margin'] = '0'
        contact_sample['@gap'] = '0'
        contact_sample['@condim'] = '4'
        contact_sample['@solimp'] = "0.99 0.99 0.01"
        contact_sample['@solref'] = "0.01 1"
        xml_hand_dict['mujoco']['contact']['pair'].append(contact_sample)

    position_sampler = PositionSampler(num_objects = num_objects)
    position_sampler.sample()


    if(objects_paras!=None):
        #use input objects parameters to generate xml files
        for i in range(num_objects):
            if(i!=0):
                dict = copy.deepcopy(xml_hand_dict['mujoco']['worldbody']['body'][-1])
                xml_hand_dict['mujoco']['worldbody']['body'].append(dict)
            dict = xml_hand_dict['mujoco']['worldbody']['body'][-1]
            geom = objects_paras[i]['type']
            shape = objects_paras[i]['size']
            pos_start = np.array([0, 0])
            quat = np.array([1,0,0,0])
            if(geom == 'capsule'):
                transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat])

            elif(geom == 'sphere'):
                transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat])
                
            elif(geom == 'box'):
                transform_start = np.concatenate([pos_start, np.array([shape[2]]), quat])
            else:
                raise NotImplementedError

            paras = objects_paras[i]
            dict['@name'] = 'object_' + str(i)
            dict['joint']['@name'] = 'object_' + str(i)
            dict['geom']['@name'] = 'object_' + str(i)
            dict['@quat'] = '{} {} {} {}'.format(transform_start[3], transform_start[4], transform_start[5], transform_start[6])
            dict['geom']['@type'] = paras['type']
            if(paras['type'] == 'sphere'):
                dict['geom']['@size'] = '{}'.format(paras['size'][0])
                dict['@pos'] = "{} {} {}".format(transform_start[0], transform_start[1], np.max([transform_start[2], paras['size'][0]]))
            elif(paras['type'] == 'box'):
                dict['geom']['@size'] = '{} {} {}'.format(paras['size'][0], paras['size'][1], paras['size'][2])
                dict['@pos'] = "{} {} {}".format(transform_start[0], transform_start[1], np.max([transform_start[2], paras['size'][2]]))
            elif(paras['type'] == 'capsule'):
                dict['geom']['@size'] = '{} {}'.format(paras['size'][0], paras['size'][1])
                dict['@pos'] = "{} {} {}".format(transform_start[0], transform_start[1], np.max([transform_start[2], paras['size'][0]]))
                dict['geom']['@quat'] = "0.707 0 0.707 0"
            else:
                raise NotImplementedError

    else:
        #generate the objects automatically
        objects_paras = []
        for i in range(num_objects):
            geoms = ['sphere', 'box', 'capsule']
            geom = geoms[np.random.randint(0,3)]
            print("geom:{}".format(geom))
            if(i!=0):
                dict = copy.deepcopy(xml_hand_dict['mujoco']['worldbody']['body'][-1])
                xml_hand_dict['mujoco']['worldbody']['body'].append(dict)
            dict = xml_hand_dict['mujoco']['worldbody']['body'][-1]

            if(geom == 'capsule'):
                theta = np.random.uniform(0, np.pi/4)
                quat = convert_glm2quat(glm.quat(np.cos(theta/2), 0, 0, np.sin(theta/2)))
                shape = np.random.uniform([0.005, 0.01], [0.01, 0.02])
                shape = np.array([shape[0], shape[1]])
                paras = {'type': geom, 'size': shape}
                pos = position_sampler.start_position[i]
                pos_start = np.array([pos[0], pos[1]])
                transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat])
                pos = position_sampler.end_position[i]
                pos_end = np.array([pos[0], pos[1], shape[0]])
                transform_end = np.concatenate([pos_end, quat])

            elif(geom == 'sphere'):
                theta = np.random.uniform(0, np.pi/4)
                quat =np.array([1, 0, 0, 0])
                shape = np.random.uniform([0.01], [0.015])
                shape = np.array([shape[0]])
                paras = {'type': geom, 'size': shape}
                pos = position_sampler.start_position[i]
                pos_start = np.array([pos[0], pos[1]])
                transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat])
                pos = position_sampler.end_position[i]
                pos_end = np.array([pos[0], pos[1], shape[0]])
                transform_end = np.concatenate([pos_end, quat])
            elif(geom == 'box'):
                theta = np.random.uniform(0, np.pi/2)
                quat = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
                shape = np.random.uniform([0.005, 0.005, 0.005], [0.01, 0.01, 0.01])
                shape = np.array([shape[0], shape[1], shape[2]])
                paras = {'type': geom, 'size': shape}
                pos = position_sampler.start_position[i]
                pos_start = np.array([pos[0], pos[1]])
                transform_start = np.concatenate([pos_start, np.array([shape[2]]), quat])
                pos = position_sampler.end_position[i]
                pos_end = np.array([pos[0], pos[1], shape[2]])
                transform_end = np.concatenate([pos_end, quat])
            else:
                raise NotImplementedError
            objects_paras.append(paras)

            dict['@name'] = 'object_' + str(i)
            dict['joint']['@name'] = 'object_' + str(i)
            dict['geom']['@name'] = 'object_' + str(i)
            dict['@quat'] = '{} {} {} {}'.format(transform_start[3], transform_start[4], transform_start[5], transform_start[6])
            dict['geom']['@type'] = paras['type']
            if(paras['type'] == 'sphere'):
                dict['geom']['@size'] = '{}'.format(paras['size'][0])
                dict['@pos'] = "{} {} {}".format(transform_start[0], transform_start[1], np.max([transform_start[2], paras['size'][0]]))
            elif(paras['type'] == 'box'):
                dict['geom']['@size'] = '{} {} {}'.format(paras['size'][0], paras['size'][1], paras['size'][2])
                dict['@pos'] = "{} {} {}".format(transform_start[0], transform_start[1], np.max([transform_start[2], paras['size'][2]]))
            elif(paras['type'] == 'capsule'):
                dict['geom']['@size'] = '{} {}'.format(paras['size'][0], paras['size'][1])
                dict['@pos'] = "{} {} {}".format(transform_start[0], transform_start[1], np.max([transform_start[2], paras['size'][0]]))
                dict['geom']['@quat'] = "0.707 0 0.707 0"
            else:
                raise NotImplementedError

    if(plate_sampler!= None):
        for i in range(plate_sampler.num_plates):
            xml_hand_dict['mujoco']['worldbody']['body'].append(copy.deepcopy(object_dict_template))
  
        for i in range(plate_sampler.num_plates):
            dict_object = xml_hand_dict['mujoco']['worldbody']['body'][-plate_sampler.num_plates + i] ########here we use 2 since the bowl is also a object
            dict_object['@name'] = 'plate_' + str(i)
            dict_object['joint']['@name'] = 'plate_' + str(i)
            dict_object['geom']['@name'] = 'plate_' + str(i)
            dict_object['@quat'] = '1 0 0 0'
            dict_object['geom']['@type'] = 'box'
           
            pos = plate_sampler.pos_list[i]
            size = plate_sampler.size_list[i]
            dict_object['geom']['@size'] = '{} {} {}'.format(size[0], size[1], size[2])
            dict_object['geom']['@density'] = '100000'
            dict_object['@pos'] = "{} {} {}".format(pos[0], pos[1], size[2])

    body = xml_hand_dict['mujoco']['worldbody']['body'][1]
    body['@user'] = dr + ' ' + dq
    for i in range(len(body['site'])):
        site = body['site'][i]
        if(site['@name'] not in ['end1', 'origin']):
            site_pos = site['@pos'].split()
            site['@pos'] = '{} {} {}'.format(float(site_pos[0])-0.0, float(site_pos[1]), float(site_pos[2]))


    body = xml_hand_dict['mujoco']['worldbody']['body'][2]
    body['@user'] = dr + ' ' + dq
    for i in range(len(body['site'])):
        site = body['site'][i]
        if(site['@name'] not in ['end2', 'origin']):
            site_pos = site['@pos'].split()
            site['@pos'] = '{} {} {}'.format(float(site_pos[0])-0.0, float(site_pos[1]), float(site_pos[2]))

    xml_hand_string = xmltodict.unparse(xml_hand_dict, pretty=True)
    #print(xml_hand_dict['mujoco']['contact']['pair'])
    #generate the xml file for simulation
    with open("./data/hand_xml_grasp/" + xml_name +  '.xml', 'w') as fd:
        fd.write(xml_hand_string)

    return objects_paras

def generate_single_task(chop_xml, pose_file, objects_paras, name, thumb_rel_pos, plate_sampler=None):
    # generate a specific task
    '''
    input:
    chop_xml: chop xml used for motion planning
    object_paras: parameters of objects
    task_name: the name of the traj file
    '''
    while(1):
        num_objects = len(objects_paras)
        transfrom_objects_start=[]
        transfrom_objects_end=[]
        for i in range(num_objects):
            geom = objects_paras[i]['type']
            shape = objects_paras[i]['size']
            pos_start = objects_paras[i]['pos_start']
            pos_end = objects_paras[i]['pos_end']
            quat_start = objects_paras[i]['quat_start']
            quat_end = objects_paras[i]['quat_end']
            
            if(geom == 'capsule'):
                transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat_start])
                transform_end = np.concatenate([pos_end, quat_end])
            elif(geom == 'sphere'):        
                transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat_start])
                transform_end = np.concatenate([pos_end, quat_end])
            elif(geom == 'box'):
                transform_start = np.concatenate([pos_start, np.array([shape[2]]), quat_start])
                transform_end = np.concatenate([pos_end, quat_end])
            else:
                raise NotImplementedError


            transfrom_objects_start.append(transform_start)
            transfrom_objects_end.append(transform_end)

        transfrom_objects_start = np.asarray(transfrom_objects_start)
        transfrom_objects_end = np.asarray(transfrom_objects_end)

        transform_chopsticks_init = np.array([0,-0.1,0.3,np.cos(-np.pi/8), 0, np.sin(-np.pi/8), 0, 0])

        traj_data = grasp_task_holder(chop_xml, pose_file, transform_chopsticks_init, objects_paras, transfrom_objects_start, transfrom_objects_end, thumb_rel_pos, name , None)
        
        if(type(traj_data) != bool):
            break
      
def generate_tasks(chop_xml, pose_file, objects_paras, num_files, save_folder, name, thumb_rel_pos, plate_sampler=None):
    #generate a batch of tasks
    num_processed_files = 0
    while(num_processed_files<num_files):
        num_objects = len(objects_paras)
        transfrom_objects_start=[]
        transfrom_objects_end=[]
        position_sampler = PositionSampler(num_objects = num_objects)
        if(plate_sampler!= None):
            position_sampler.sample(plate_sampler)
        else:
            position_sampler.sample()
        for i in range(num_objects):
            geom = objects_paras[i]['type']
            shape = objects_paras[i]['size']
            
            if(geom == 'capsule'):
                theta = np.random.uniform(0, np.pi/4)
                quat = convert_glm2quat(glm.quat(np.cos(theta/2), 0, 0, np.sin(theta/2)))
                pos = position_sampler.start_position[i]
                pos_start = np.array([pos[0], pos[1]])
                if(pos[2] == 0):
                    transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat])
                else:
                    transform_start = np.concatenate([pos_start, np.array([shape[0] + pos[2]]), quat])
                pos = position_sampler.end_position[i]
                if(pos[2] == 0):
                    pos_end = np.array([pos[0], pos[1], shape[0]])
                else:
                    pos_end = np.array([pos[0], pos[1], shape[0] + pos[2]])
                transform_end = np.concatenate([pos_end, quat])
            elif(geom == 'sphere'):
                theta = np.random.uniform(0, np.pi/4)
                quat =np.array([1, 0, 0, 0])
                pos = position_sampler.start_position[i]
                pos_start = np.array([pos[0], pos[1]])
                if(pos[2] == 0):
                    transform_start = np.concatenate([pos_start, np.array([shape[0]]), quat])
                else:
                    transform_start = np.concatenate([pos_start, np.array([shape[0] + pos[2]]), quat])
                pos = position_sampler.end_position[i]
                if(pos[2] == 0):
                    pos_end = np.array([pos[0], pos[1], shape[0]])
                else:
                    pos_end = np.array([pos[0], pos[1], shape[0] + pos[2]])
                transform_end = np.concatenate([pos_end, quat])
            elif(geom == 'box'):
                theta = np.random.uniform(0, np.pi/2)
                quat = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
                pos = position_sampler.start_position[i]
                pos_start = np.array([pos[0], pos[1]])
                if(pos[2]== 0):
                    transform_start = np.concatenate([pos_start, np.array([shape[2]]), quat])
                else:
                    transform_start = np.concatenate([pos_start, np.array([shape[2] + pos[2]]), quat])
                pos = position_sampler.end_position[i]
                if(pos[2] == 0):
                    pos_end = np.array([pos[0], pos[1], shape[2]])
                else:
                    pos_end = np.array([pos[0], pos[1], shape[2] + pos[2]])
                transform_end = np.concatenate([pos_end, quat])
            else:
                raise NotImplementedError

            transfrom_objects_start.append(transform_start)
            transfrom_objects_end.append(transform_end)

        transfrom_objects_start = np.asarray(transfrom_objects_start)
        transfrom_objects_end = np.asarray(transfrom_objects_end)

        transform_chopsticks_init = np.array([0,-0.1,0.3,np.cos(-np.pi/8), 0, np.sin(-np.pi/8), 0, 0])
      
        traj_data = grasp_task_holder(chop_xml, pose_file, transform_chopsticks_init, objects_paras, transfrom_objects_start, transfrom_objects_end, thumb_rel_pos, save_folder, name, plate_sampler)
        
        if(type(traj_data) == bool):
            pass
        else:
            num_processed_files += 1

        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chop_xml", type=str)
    parser.add_argument("--hand_xml", type=str)
    parser.add_argument("--pose", type = str)
    parser.add_argument("--task_name", type=str)
    args = parser.parse_args()

    if(not os.path.exists("./data/" + args.task_name)):
        os.mkdir("./data/" + args.task_name)
        

    objects_paras = []
    objects_paras.append({'type': 'sphere', "size": np.array([0.01])})
    objects_paras.append({'type': 'capsule', "size": np.array([0.005, 0.01])})
    objects_paras.append({'type':'box' , "size": np.array([0.005, 0.005, 0.005])})

    generate_xml(3, args.chop_xml, args.hand_xml, args.task_name, objects_paras)
    print('start generation')
    for j in range(1):
        print('No.{}'.format(j))
        generate_tasks(args.chop_xml, args.pose, objects_paras, 1,  args.task_name, str(j), 0)
    