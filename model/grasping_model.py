import time

import mujoco_py
import numpy as np
import pytorch3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_invert, quaternion_raw_multiply, quaternion_to_matrix
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from model.network import ConfigurationNet, ReachNet

#some util functions used for grasping prediction
def quat2mat(x):
    #convert a quaternion to a 6d rotation representation
    '''
    input: quaternion (w,x,y,z)
    output: a 6-d rotation representation
    '''
    r = R.from_quat([x[1], x[2], x[3], x[0]])
    x = r.as_matrix()
    x = x[0:2].reshape((-1))
    return x

def grasp2category(x):
    phi = x[0]
    theta = x[1]
    angle = x[2]
    num_phi = 10
    num_theta = 20
    num_angle = 10
    i = int(np.floor(phi/(np.pi/10)))
    j = int(np.floor(theta/(2 * np.pi/20)))
    k = int(np.floor((angle + np.pi)/(2 * np.pi/10)))
    label = np.zeros((num_phi * num_theta * num_angle, ))
    label[i * (num_theta * num_angle) + j * (num_angle) + k] = 1
    return np.array([i * (num_theta * num_angle) + j * (num_angle) + k])

def category2grasp(x):
    num_theta = 20
    num_angle = 10
    idx = x
    i = np.asarray(idx / (num_theta * num_angle), dtype=np.int32)
    j = np.asarray((idx - i * (num_theta * num_angle))/(num_angle), dtype=np.int32)
    k = idx - i * (num_theta * num_angle) - j * num_angle
    phi = i * np.pi /10
    theta = j * np.pi *2 /20
    angle = -np.pi + 2 * np.pi/10 *k
    return np.concatenate([phi, theta, angle], 1)

def T(x):
    '''
    convert numpy data into tensor type
    '''
    return torch.Tensor(x)

def quat2grasp(q):
    r = np.zeros((q.shape[0], 4))
    r[:,1] = -1
    ori = quaternion_raw_multiply( quaternion_raw_multiply(T(q), T(r)) ,  quaternion_invert(T(q))).numpy()
    phi = np.arccos(ori[:, 3])
    theta = np.arccos(ori[:, 1]/np.sin(phi))
    theta = (ori[:,2]>=0) * theta + (ori[:,2] <0) * (np.pi *2 - theta)

    vec1 = np.zeros((q.shape[0],3))
    vec1[:,0] = -1
    vec2 = np.zeros((q.shape[0], 3))
    vec2[:,0] = ori[:,1]
    vec2[:,1] = ori[:,2]
    vec2[:,2] = ori[:,3]

    axis = np.cross(vec1, vec2)
    angle = np.arccos(np.diag(np.dot(vec1, vec2.T)))
    axis = axis/np.sqrt((axis**2).sum(1)).reshape((-1,1))
    q0 = np.zeros((q.shape[0], 4))
    q0[:,0] = np.cos(angle/2)
    q0[:,1] = axis[:,0] * np.sin(angle/2)
    q0[:,2] = axis[:,1] * np.sin(angle/2)
    q0[:,3] = axis[:,2] * np.sin(angle/2)
    dq = quaternion_raw_multiply(quaternion_invert(T(q0)), T(q)).numpy() 
    dq = (dq[:,0]>0).reshape((-1,1)) * dq + (dq[:,0] <=0).reshape((-1,1)) * dq * -1
    
    #print('dq:{}'.format(dq))
    rot =  np.arcsin(np.clip(-dq[:,1], -1, 1)) * 2
    res = np.zeros((q.shape[0], 3))
    res[:,0] = phi
    res[:,1] = theta
    res[:,2] = rot
    return res

def grasp2quat(x):
    rot = x[:, 2]
    phi = x[:, 0]- np.pi/2
    theta = x[:, 1]

    qx = np.zeros((x.shape[0], 4))
    qx[:,0] = np.cos(phi/2)
    qx[:,2] = -np.sin(phi/2)

    qy = np.zeros((x.shape[0], 4))
    qy[:,0] = np.cos(theta/2)
    qy[:,3] = np.sin(theta/2)

    dq = np.zeros((x.shape[0], 4))
    dq[:,0] = np.cos(rot/2)
    dq[:,1] = -1 * np.sin(rot/2)
    q = pytorch3d.transforms.quaternion_raw_multiply(torch.Tensor(qy), torch.Tensor(qx))
    q = pytorch3d.transforms.quaternion_raw_multiply(q, torch.Tensor(dq)).numpy()
    return q


def quat2q(kin_chopsticks, x, object_pos = np.array([0,0,0]), object_paras = np.array([0.01, 0.01, 0.01])):
    q = x
    _, pos_end2, vec_end = kin_chopsticks.fk(np.array([0,0,0,q[0],q[1],q[2],q[3],0]))
    pos_end2_target = object_pos + vec_end * (object_paras[0] + 0.0075)
    pos_chopsticks = pos_end2_target - pos_end2

    def f(x):
        x = np.concatenate([pos_chopsticks, np.array([q[0], q[1], q[2], q[3]]), x])
        pos_end1,pos_end2,_ = kin_chopsticks.fk(x)
        center_end = 0.5 * (pos_end1 + pos_end2)
        r = np.linalg.norm(pos_end1 - object_pos) - (object_paras[0] + 0.01)
        loss = 100 * r **2 + np.exp(-1000 * np.min([0, r])) - 1
        return loss
   
    bound = np.array([[-0.2,0.1]]) #shape (N,2)
    bound = list(zip(*(bound.T)))
    init_pos = np.array([-0.2])
    res = minimize(f, init_pos, bounds= bound, method="L-BFGS-B")
    print('pose quality:{}'.format(res.fun))
    q = np.concatenate([pos_chopsticks, np.array([q[0], q[1], q[2], q[3]]), res.x])
    return q

def grasp2q_coarse(x, object_closure):
    x = grasp2quat(x)
 
    r = np.zeros((x.shape[0],4))
    r[:,1] = -0.13 #half length of the chopsticks
    pos = quaternion_raw_multiply(T(x), T(r))
    pos = quaternion_raw_multiply(pos, quaternion_invert(T(x))).numpy()
    
    r = np.zeros((x.shape[0],4))
    r[:,2] = 1 #vec: from the end2 to end1
 
    dir = quaternion_raw_multiply(T(x), T(r))
    dir = quaternion_raw_multiply(dir, quaternion_invert(T(x))).numpy()

    pos = -pos - dir * object_closure
    pos = pos[:,1:]
    return np.concatenate([pos, x], 1)


def transform_global(x, transform_object):
    '''
    transform the grasp parameter and reach parameter into global space
    x: the input grasp or reach parameter
    transform_object: the transformation of the object
    y:
    '''
    if(x.shape[1] == 4):
        #grasp parameter
        q_global = transform_object[:,3:]
        return quaternion_raw_multiply(T(q_global), T(x)).numpy()
    elif(x.shape[1] == 7):
        #reach parameter
        q_global = transform_object[:,3:]
        q =  quaternion_raw_multiply(T(q_global), T(x[:,3:])).numpy()
        pos_tensor = np.zeros((x.shape[0], 4))
        pos_tensor[:,1:] = x[:,0:3].copy()
        pos_tensor = quaternion_raw_multiply( quaternion_raw_multiply(T(q_global), T(pos_tensor)) ,  quaternion_invert(T(q_global))).numpy()
        pos = pos_tensor[:,1:4] + transform_object[:,0:3]
        return np.concatenate([pos, q], 1)
    else:
        raise NotImplementedError


def quat2mat_batch(x):
    '''
    convert the quaternion into 6d representation
    batch operation
    x: the input quaternions (N, 4)
    '''
    x = T(x).float()
    mat = quaternion_to_matrix(x).numpy()
    mat = mat[:,0:2] #(N, 2,3)
    mat = mat.reshape((x.shape[0], -1))
    return mat


class kinematics_chopsticks(object):
    '''
    the class of the kinematic chopsticks: without the arm
    '''
    def __init__(self) -> None:
        super().__init__()
        model = mujoco_py.load_model_from_path('./data/chopsticks_xml/chopsticks.xml')
        self.sim = mujoco_py.MjSim(model)
        self.site_end1 = self.sim.model.site_name2id("end1")
        self.site_end2 = self.sim.model.site_name2id("end2")
    
    def fk(self,q):
        '''
        forawrd kinematics: compute positions of two endpoints and the vector pointing from the end2 to end1
        '''
        self.sim.data.qpos[0:8] = q
        self.sim.forward() 
        pos_end1 = self.sim.data.site_xpos[self.site_end1]
        pos_end2 = self.sim.data.site_xpos[self.site_end2]
        vec_end = pos_end2 - pos_end1
        vec_end = vec_end/(np.linalg.norm(vec_end))
        return pos_end1, pos_end2, vec_end


############################## The class of the grasping model ############################################


class GraspingModel(object):

    def __init__(self, grasp_path, reach_path) -> None:
        self.grasp = ConfigurationNet(4, 2000, [512, 512])
        self.reach = ReachNet(9, 1, [256, 256])
        self.grasp.load_state_dict(torch.load(grasp_path))
        self.reach.load_state_dict(torch.load(reach_path))
        self.grasp_para = category2grasp(np.arange(2000).reshape((-1,1))) #convert the label to spherical grasping representation
        self.quat_para = grasp2quat(self.grasp_para)
        self.num_candidates = 20 #top k selection
        self.kin_chopsticks = kinematics_chopsticks()

    def inference(self, x, object_transformation, chopsticks_transformation):
        '''
        x:(1,-1)
        '''
        paras_object = x[0,-3:]
        x = T(x[:,0:4]).float()
        #start = time.time()
        with torch.no_grad():
            grasp_prob = F.softmax(self.grasp(x).view((-1,1)), dim= 0) 
        self.reach_para = grasp2q_coarse(self.grasp_para, np.max([paras_object[0], paras_object[1], paras_object[2]]))
        reach_para = transform_global(self.reach_para, object_transformation) #(N = 500, 7)
        reach_para_6drot = quat2mat_batch(reach_para[:,3:])
        reach_para = np.concatenate([reach_para[:,0:3], reach_para_6drot], 1)
        quat_para = transform_global(self.quat_para, object_transformation) #(N = 500, 3)

        diff_quat = quaternion_raw_multiply(quaternion_invert(T(chopsticks_transformation[:,3:])), T(quat_para)).numpy()
        diff_pos = np.sqrt(((chopsticks_transformation[:,0:3] - reach_para[:,0:3])**2).sum(1))
        diff_quat = np.arccos(np.clip(diff_quat[:,0],-1,1)) * 2
        nature_score = (0. * torch.exp(-10 * T(diff_pos)) + 1.0 * np.exp(-5 * T(diff_quat))).view((-1,1))
        over_score = grasp_prob * nature_score
        idx_filter = over_score.argsort(0).view(-1)[-50:]
        over_score_filter = over_score[idx_filter[-50:]]
        reach_para_filter = reach_para[idx_filter[-50:].numpy()]
        reach_score_filter = self.reach(T(reach_para_filter).float()) #(N, 1)
        over_score = over_score_filter *  reach_score_filter
      
        idx_new = over_score.argsort(0)[-1,0].item()
        idx = idx_filter[idx_new].item()
        quat = quat_para[idx,:]
        if(x[0,0]==1 or x[0,2] == 1):
            #print(self.grasp_para[idx])
            q = quat2q(self.kin_chopsticks, quat, object_transformation[0,0:3], paras_object)
        elif(x[0,1] == 1):
            theta = self.grasp_para[idx, 1]
            if(abs(theta - 0)<1e-3 or abs(theta - np.pi)<1e-3):
                paras_object[0] = paras_object[1]
            else:
                pass
            q = quat2q(self.kin_chopsticks, quat, object_transformation[0,0:3], paras_object)
        else:
            raise NotImplementedError
        return q


if __name__ == "__main__":
    #train the grasping neural network
    # train_grasp_network('./data/grasp/training_input.npy', './data/grasp/training_label.npy')
    #test our grasp network and reach network
    model = GraspingModel('./data/grasp/graspnet_mlp.pth', './data/reach/reachnet_mlp.pth')
    x = np.array([[0,0,1,0,0.01,0.02,0.02]])
    start = time.time()
    q = model.inference(x, np.array([[-0.,0.,0.01,np.cos(0.5/2),0,0,np.sin(-0.5/2)]]), np.array([[0,-0.1,0.2,np.cos(-np.pi/8), 0, np.sin(-np.pi/8), 0]]))
    print(q)
    end = time.time()
    print('dt:{}'.format(end - start))
    model = mujoco_py.load_model_from_path('./data/chopsticks_xml/chopsticks.xml')
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    while(1):
        sim.data.qpos[0:8] = q.copy()
        sim.forward()
        viewer.render()
    