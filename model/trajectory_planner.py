import math
import time

import mujoco_py
from IPython import embed
from scipy.optimize import minimize
from scipy.stats import beta
from utils.convert_util import *


def beizer(p0, p1, p2, t):
    '''
    second-order beizier curves
    '''
    res =   (1-t)**2 * p0 + 2 * t * (1-t) * p1 + t**2 * p2
    if(math.isnan(res[0])):
        embed()
    return res

def clog(x):
    '''
    the log barrier function 
    '''
    if(x>0.01):
        return 0
    else:
        return -(x-0.01)**2/(x) * np.log(x/0.01)

def compute_transform_object_from_chopsticks(transform_chopsticks, dr, dq):
        '''
        given transform of the chopsticks, dr and dq, compute the transformation of the object
        '''
        q = convert_glm2quat(convert_quat2glm(transform_chopsticks[3:7]) * dq)
        pos = transform_chopsticks[0:3] + convert_glm2pos(convert_quat2glm(transform_chopsticks[3:7]) * dr * glm.inverse(convert_quat2glm(transform_chopsticks[3:7])))
        return np.concatenate([pos,q]) 


class TrajectoryPlanner(object):
    '''
    compute collision-free chopsticks trajectories 
    '''
    def __init__(self, xml, start_dof,end_dof, linear_v = 0.3, ang_v = np.pi*0.4) -> None:
        super().__init__()
        self.start_dof = start_dof
        self.end_dof = end_dof
        self.linear_v = linear_v
        self.ang_v = ang_v

        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
        self.geom_chopsticks = [self.sim.model.geom_name2id('chopstick2'), self.sim.model.geom_name2id('chopstick1')]
        self.geom_palm = self.sim.model.geom_name2id('palm_virtual')


    def opt_move_seq(self, q_init = None, start_transformation = np.array([0,0,0,1,0,0,0]), end_transformation=np.array([0,0,0,1,0,0,0]), object_idx = 0, dr = None, dq = None, mode = 'pregrasp'):
        '''
        given the xml file describing the scene configuration, start and end transformations of the chopsticks,
        compute a collision-free trajectories of the chopsticks
        '''
     
        object_name = 'object_' + str(object_idx)
        geom_object = self.sim.model.geom_name2id(object_name)
        if(mode == 'pregrasp'):
            geom_object = -100 # for pregrasp mode, we do not want the chopsticks collide with the object
        self.sim.data.qpos[self.start_dof:self.end_dof] = q_init.copy()
        self.sim.forward()
        t_move = np.linalg.norm(start_transformation[0:3] - end_transformation[0:3])/self.linear_v
        t_rotate = glm.inverse(convert_quat2glm(start_transformation[3:7])) * convert_quat2glm(end_transformation[3:7])
        if(t_rotate[3]<0):
            t_rotate = -t_rotate
        t_rotate = np.arccos(np.clip(t_rotate[3], -1, 1)) * 2/self.ang_v
        t = np.max([t_move, t_rotate])
        t = np.max([t, 0.5]) # to restrict too short motions
        t = np.floor(t/0.1) * 0.1
        print("t:{}".format(t))

        if(t >20):
            embed()

        beta_para1 = 1
        beta_para2 = 1


        def f(x, compute_dist = 1):
            #start = time.time()
            '''
            obj functions where the control middle control points are optimized
            '''
            loss = 0
            dist = 0
            dt = 0.01
            for i in range(int(np.floor(t/dt)) + 1):
                pos = beizer(start_transformation[0:3], x , end_transformation[0:3], i*dt/t)
                dist += np.linalg.norm(beizer(start_transformation[0:3], x , end_transformation[0:3], i*dt/t) - beizer(start_transformation[0:3], x , end_transformation[0:3], (i+1)*dt/t))
                q = glm.slerp(convert_quat2glm(start_transformation[3:7]), convert_quat2glm(end_transformation[3:7]), beta.cdf(i*dt/t, beta_para1, beta_para2))
                self.sim.data.qpos[0:3] = pos
                self.sim.data.qpos[3:7] = np.array([q[3],q[0],q[1],q[2]])
                self.sim.data.qpos[7] = start_transformation[-1] + (end_transformation[-1] - start_transformation[-1]) * (i*dt/t)
                if(mode!='pregrasp'):
                    transform_chopsticks = self.sim.data.qpos[0:8].copy()
                    q_object = compute_transform_object_from_chopsticks(transform_chopsticks, dr, dq)
                    self.sim.data.qpos[8 + 7 * object_idx: 8 + 7 * object_idx + 7] = q_object
                self.sim.forward()
                for j in range(self.sim.data.ncon):
                    contact = self.sim.data.contact[j]
                    if((contact.geom2 in self.geom_chopsticks and contact.geom1 != geom_object)):
                        dis_contact = contact.dist
                        loss += np.min([np.exp(1e4 * abs(dis_contact)), 1e6]) - 1
                    if((contact.geom1 in self.geom_chopsticks and contact.geom2 != geom_object)):
                        dis_contact = contact.dist
                        loss += np.min([np.exp(1e4 * abs(dis_contact)), 1e6]) - 1
                    if((contact.geom2 not in self.geom_chopsticks and contact.geom1 == geom_object)):
                        if(mode == 'moving'):
                            dis_contact = contact.dist
                            loss += np.exp(1e4 * abs(dis_contact)) - 1
                    if((contact.geom1 not in self.geom_chopsticks and contact.geom2 == geom_object)):
                        if(mode == 'moving'):
                            dis_contact = contact.dist
                            loss += np.exp(1e4 * abs(dis_contact)) - 1
                    if((contact.geom1 == self.geom_palm)):
                        dis_contact = contact.dist
                        loss += np.exp(1e4 * abs(dis_contact)) - 1
                    if((contact.geom2 == self.geom_palm)):
                        dis_contact = contact.dist
                        loss += np.exp(1e4 * abs(dis_contact)) - 1

            loss += compute_dist * (dist)
            return loss

        bound = np.array([[-0.4,0.4],[-0.4,0.4],[0,1]])
        bound = list(zip(*(bound.T)))
        init_pos = 0.5 * (start_transformation[0:3] + end_transformation[0:3])
        init_pos[2] += 0.075
       

       
        fun = 100
        num =  0
        start = time.time()
        while(fun >1e-4):
            init_pos_sample = init_pos + np.random.uniform([-0.25,-0.25,0.], [0.25,0.25,0.2])
            if(f(init_pos_sample.copy(),0)>0 and num < 100):
                #print(f(init_pos_sample.copy(),0))
                num += 1
                continue
            else:
                if(num >= 100):
                    return False 
                res = minimize(f, init_pos_sample, bounds= bound, method="L-BFGS-B")
                fun = res.fun
                print('opt')
                print("original value:{}".format(f(init_pos_sample, 0)))
                print(fun)
                x = res.x # the posiitons of the middle control points
                break
        end = time.time()
        print('traj opt dt:{}'.format(end - start))

        #rescale the velcotity
        dist = f(x, 1)
        if(dist>100):
            embed()
        t = np.max([dist/self.linear_v, t])

        motions = np.zeros((int(np.floor(t/0.01)) + 1, 1 + 8))
        motions[:-1,0] = 0.01
        for i in range(motions.shape[0]):
            pos = beizer(start_transformation[0:3], x, end_transformation[0:3], i*0.01/t)
            q = glm.slerp(convert_quat2glm(start_transformation[3:7]), convert_quat2glm(end_transformation[3:7]), beta.cdf(i*0.01/t, beta_para1, beta_para2))
            if(end_transformation[3] * q[3] <0):
                q = -q
            theta = start_transformation[-1] + (end_transformation[-1] - start_transformation[-1]) * (i*0.01/t)
            pose = np.concatenate([pos,convert_glm2quat(q), np.array([theta])])
            motions[i, 1:] = pose
        return motions
