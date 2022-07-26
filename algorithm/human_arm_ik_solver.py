from IPython import embed
import math       
import xmltodict
import numpy as np
from mujoco_py import load_model_from_xml, MjSim
import glm
import time
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from utils.convert_util import *

def compute_transform_object_from_chopsticks(transform_chopsticks, dr, dq):
        '''
        given transform of the chopsticks, dr and dq, compute the transformation of the object
        '''
        q = convert_glm2quat(convert_quat2glm(transform_chopsticks[3:7]) * dq)
        pos = transform_chopsticks[0:3] + convert_glm2pos(convert_quat2glm(transform_chopsticks[3:7]) * dr * glm.inverse(convert_quat2glm(transform_chopsticks[3:7])))
        return np.concatenate([pos,q]) 
# the class of the human arm IK solver
class HumanArm_IK_Solver(object):
    '''
    the class of the 7 dof human arm inverse kinematic algorithm
    in order to solve the joint redundancy problem, we use linear sensor motor model to eliminatre the redundant solutions
    '''
    def __init__(self, xml_path) -> None:
        super().__init__()
        model = load_model_from_xml(xml_path)
        xml_dict = xmltodict.parse(xml_path)
        body = xml_dict['mujoco']['worldbody']['body'][0]
        while(body['@name'] != 'chopstick2'):
            body = body['body']
        pos_str = body['@pos'].split()
        quat_str = body['@quat'].split()
        self.dr_chopstick2 = glm.quat(0, float(pos_str[0]), float(pos_str[1]), float(pos_str[2]))
        self.dq_chopstick2 = glm.quat(float(quat_str[0]), float(quat_str[1]), float(quat_str[2]), float(quat_str[3]))
        
        self.sim = MjSim(model)
        self.sim.forward()
        self.pos_shoulder = self.sim.data.body_xpos[self.sim.model.body_name2id('upperarm')].copy() #compute the position of the shoulder
        self.jnt_range = self.sim.model.jnt_range[0:7].copy()
        self.geom_chopsticks = [self.sim.model.geom_name2id('chopstick2'), self.sim.model.geom_name2id('chopstick1')]
        

    def compute_target_transformation(self, transformation):
        '''
        given the target transformation of the chopsticks
        compute the desirable transformation of the wrist
        '''
        target_q = glm.quat(transformation[3], transformation[4], transformation[5], transformation[6])
        target_pos = glm.quat(0, transformation[0], transformation[1], transformation[2])
        #compute the target position and  rotation of the wrist
        q_wrist = target_q * glm.inverse(self.dq_chopstick2)
        if(math.isnan(q_wrist[3])):
            embed()
        pos_wrist = target_pos - q_wrist * (self.dr_chopstick2 + glm.quat(0, 0, 0, 0.034)) * glm.inverse(q_wrist)
        return convert_glm2pos(pos_wrist), convert_glm2quat(q_wrist)

    def linear_model(self, pos_wrist):
        '''
        given target transformation of the wrist2 body
        compute the target position of the elbow joint
        '''
        pos_wrist_rel = pos_wrist - self.pos_shoulder
        R = np.linalg.norm(pos_wrist_rel) * 100
        X = np.arctan(pos_wrist_rel[0]/pos_wrist_rel[1]) * 180 /np.pi
        phi = np.arctan(pos_wrist_rel[2]/np.linalg.norm(pos_wrist_rel[0:2])) * 180 /np.pi
        theta = -4 + 1.10 * R + 0.9 * phi
        beta = 39.4 + 0.54 * R -1.06 * phi
        ri = 13.2 + 0.86 * X + 0.11 * phi
        alpha = -10 + 1.08 * X -0.35 * phi

        x = 0.3 * np.sin(theta * np.pi /180) * np.sin(ri * np.pi /180)
        y = 0.3 * np.sin(theta * np.pi /180) * np.cos(ri * np.pi /180)
        z = -0.3 * np.cos(theta * np.pi /180)

        x1 = 0.3 * np.sin(theta * np.pi /180) * np.sin(ri * np.pi /180) + 0.3 * np.sin(beta * np.pi /180) * np.sin(alpha * np.pi /180)
        y1 = 0.3 * np.sin(theta * np.pi /180) * np.cos(ri * np.pi /180) + 0.3 * np.sin(beta * np.pi /180) * np.cos(alpha * np.pi /180)
        z1 = -0.3 * np.cos(theta * np.pi /180) + 0.3 * np.cos(beta * np.pi / 180)

        pos_elbow_approx = np.array([x,y,z])
        pos_wrist_approx = np.array([x1,y1,z1])
        return pos_elbow_approx, pos_wrist_approx

    def ik_first4(self, pos_elbow_ref, pos_wrist_ref):
        '''
        given the position of the elbow, compute the first 4 joint angles
        '''
        #compute q1,q2,q4
        q1 = np.arctan(-pos_elbow_ref[1]/pos_elbow_ref[2])
        q2 = np.arctan(-pos_elbow_ref[0]/np.sqrt(pos_elbow_ref[1]**2 + pos_elbow_ref[2]**2))
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist_ref)**2)/(2 * 0.3 * 0.3), -1, 1))
        #compute q3
        q_temp = glm.quat(np.cos(q1/2), np.sin(q1/2), 0, 0) * glm.quat(np.cos(q2/2), 0 , np.sin(q2/2), 0) * glm.quat(np.cos(q4/2), np.sin(q4/2), 0, 0)
        pos_wrist_init = pos_elbow_ref + convert_glm2pos(q_temp * glm.quat(0, 0, 0.3, 0) * glm.inverse(q_temp))
        axis = pos_elbow_ref.copy()/np.linalg.norm(pos_elbow_ref)
        vec_wrist_init = pos_wrist_init - axis * (np.dot(axis, pos_wrist_init))
        vec_wrist_end = pos_wrist_ref - axis * (np.dot(axis, pos_wrist_ref))
        q3 = np.dot(vec_wrist_init, vec_wrist_end)/(np.linalg.norm(vec_wrist_init) * np.linalg.norm(vec_wrist_end))
        q3 = np.arccos(np.clip(q3,-1,1))
        if(np.dot(np.cross(vec_wrist_init, vec_wrist_end), axis)<0):
            q3 = q3
        else:
            q3 = -q3
        q = np.array([q1,q2,q3,q4])
        return q
    
    def ik_last3(self, q_wrist_ref, q_first4):
        '''
        given target quaterion of the wrist and angles of the first 4 joints,
        compute the last 3 joint angles
        '''
        q1,q2,q3,q4 = q_first4[0], q_first4[1], q_first4[2], q_first4[3]
        q_temp = glm.quat(np.cos(-np.pi/4), np.sin(-np.pi/4), 0, 0) * glm.quat(np.cos(q1/2), np.sin(q1/2), 0, 0) * glm.quat(np.cos(q2/2), 0 , 0, np.sin(q2/2)) *  glm.quat(np.cos(q3/2), 0 , np.sin(-q3/2), 0) * glm.quat(np.cos(q4/2), np.sin(q4/2), 0, 0)
        dq = glm.inverse(q_temp) * convert_quat2glm(q_wrist_ref)
        if(math.isnan(q_wrist_ref[0])):
            embed()
        dq = R.from_quat([dq[0], dq[1], dq[2], dq[3]])
        euler_angles = dq.as_euler('XYZ')
        return euler_angles

    def ik_first3_fromlast4(self, q_wrist_ref, q_last4):
        '''
        compute the first 3 joint angles given the last 4 joint angles
        '''
        q4,q5,q6,q7 = q_last4[0], q_last4[1], q_last4[2], q_last4[3]
        q_temp = glm.quat(np.cos(q4/2), np.sin(q4/2),0,0) * glm.quat(np.cos(q5/2), np.sin(q5/2), 0, 0) * glm.quat(np.cos(q6/2), 0, np.sin(q6/2),0) * glm.quat(np.cos(q7/2), 0, 0, np.sin(q7/2))
        dq = convert_quat2glm(q_wrist_ref) * glm.inverse(q_temp)

        q_shoulder = glm.inverse(glm.quat(np.cos(-np.pi/4), np.sin(-np.pi/4), 0, 0)) * dq
        q_shoulder = R.from_quat([q_shoulder[0], q_shoulder[1], q_shoulder[2], q_shoulder[3]]).as_euler('XZY')
        q_shoulder[2] = - q_shoulder[2]
        return q_shoulder




    def sol_1(self, pos_wrist,quat_wrist, q):
        '''
        generate the first solution , the first joint angle is fixed
        pos_wirst,: the position of the wirst
        quat_wrist, the quaternion of the wrist
        q1: joint angle of the first joint
        '''
        q1 = q[0]
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist)**2)/(2 * 0.3 * 0.3), -1, 1))
        #the first solition
        #print((pos_wrist[1] * np.cos(q1) + pos_wrist[2] * np.sin(q1))/ (0.3 * np.cos(q4)))
        q3 = -np.arccos(np.clip((pos_wrist[1] * np.cos(q1) + pos_wrist[2] * np.sin(q1))/ (0.3 * np.cos(q4)),-1,1))#range [-1.57, 0]

        #no feasiable solution
        if(abs((pos_wrist[1] * np.cos(q1) + pos_wrist[2] * np.sin(q1))/ (0.3 * np.cos(q4)))>1):
            return np.zeros((7,))

        A11 = -0.3 * np.sin(q3) * np.cos(q4)
        A12 = -0.3 + 0.3 * np.sin(q4)

        A = np.array([[A11,A12],[-A12,A11]])
        b = np.array([[pos_wrist[0]],[-pos_wrist[2] * np.cos(q1) + pos_wrist[1] * np.sin(q1)]])
        sol = np.dot(np.linalg.inv(A), b)
        q2 = np.arctan2(sol[1, 0], sol[0, 0])
        q_first4 = np.array([q1,q2,q3,q4]).copy()
    
        q_last3 = self.ik_last3(quat_wrist, q_first4)
        sol1 = np.concatenate([q_first4, q_last3]).copy()
        #the second solution
        q3 = np.arccos(np.clip((pos_wrist[1] * np.cos(q1) + pos_wrist[2] * np.sin(q1))/ (0.3 * np.cos(q4)),-1,1))
        A11 = -0.3 * np.sin(q3) * np.cos(q4)
        A12 = -0.3 + 0.3 * np.sin(q4)

        A = np.array([[A11,A12],[-A12,A11]])
        b = np.array([[pos_wrist[0]],[-pos_wrist[2] * np.cos(q1) + pos_wrist[1] * np.sin(q1)]])
        sol = np.dot(np.linalg.inv(A), b)
        q2 = np.arctan2(sol[1, 0], sol[0, 0])
        q_first4 = np.array([q1,q2,q3,q4]).copy()
        q_last3 = self.ik_last3(quat_wrist, q_first4)
        sol2 = np.concatenate([q_first4, q_last3]).copy()

        if(np.linalg.norm(q -sol1) > np.linalg.norm(q - sol2)):
            sol = sol2
        else:
            sol = sol
        return sol

    def sol_2(self, pos_wrist,quat_wrist, q):
        '''
        generate the second solution , the second joint angle is fixed
        pos_wirst,: the position of the wirst
        quat_wrist, the quaternion of the wrist
        q1: joint angle of the first joint
        '''
        q2 = q[1]
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist)**2)/(2 * 0.3 * 0.3), -1, 1))
        #the first solition
        q3 = np.arcsin(np.clip((pos_wrist[0] - (-0.3 + 0.3 * np.sin(q4)) * np.sin(q2))/(-0.3 * np.cos(q4)* np.cos(q2)), -1, 1))

        if(abs((pos_wrist[0] - (-0.3 + 0.3 * np.sin(q4)) * np.sin(q2))/(-0.3 * np.cos(q4)* np.cos(q2)))>1):
            return np.zeros((7, ))

        A = np.array([[pos_wrist[1], pos_wrist[2]], [-pos_wrist[2], pos_wrist[1]]])
        b = np.array([[0.3 * np.cos(q4) * np.cos(q3)], [(0.3 - 0.3 * np.sin(q4)) * np.cos(q2) - 0.3 * np.cos(q4) * np.sin(q3) * np.sin(q2)]])
        sol = np.dot(np.linalg.inv(A), b)
        q1 = np.arctan2(sol[1, 0], sol[0, 0])
        q_first4 = np.array([q1,q2,q3,q4]).copy()
        q_last3 = self.ik_last3(quat_wrist, q_first4)
        sol1 = np.concatenate([q_first4, q_last3]).copy()
        #the second solution
        q3 = np.pi - np.arcsin(np.clip((pos_wrist[0] - (-0.3 + 0.3 * np.sin(q4)) * np.sin(q2))/(-0.3 * np.cos(q4)* np.cos(q2)), -1, 1))
        A = np.array([[pos_wrist[1], pos_wrist[2]], [-pos_wrist[2], pos_wrist[1]]])
        b = np.array([[0.3 * np.cos(q4) * np.cos(q3)], [(0.3 - 0.3 * np.sin(q4)) * np.cos(q2) - 0.3 * np.cos(q4) * np.sin(q3) * np.sin(q2)]])
        sol = np.dot(np.linalg.inv(A), b)
        q1 = np.arctan2(sol[1, 0], sol[0, 0])
        q_first4 = np.array([q1,q2,q3,q4]).copy()
        q_last3 = self.ik_last3(quat_wrist, q_first4)
        sol2 = np.concatenate([q_first4, q_last3]).copy()

        if(np.linalg.norm(q -sol1) > np.linalg.norm(q - sol2)):
            sol = sol2
        else:
            sol = sol1
       
        return sol

    def sol_3(self, pos_wrist,quat_wrist, q):
        '''
        generate the third solution , the third joint angle is fixed
        pos_wirst,: the position of the wirst
        quat_wrist, the quaternion of the wrist
        q1: joint angle of the first joint
        '''
        q3 = q[2]
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist)**2)/(2 * 0.3 * 0.3), -1, 1))
        #the first solition
        q1 = np.arcsin(-0.3 * np.cos(q3) * np.cos(q4)/np.sqrt(pos_wrist[1]**2 + pos_wrist[2]**2)) - np.arctan(pos_wrist[1]/pos_wrist[2])
        if(abs(-0.3 * np.cos(q3) * np.cos(q4)/np.sqrt(pos_wrist[1]**2 + pos_wrist[2]**2)) - np.arctan(pos_wrist[1]/pos_wrist[2])>1):
            return np.zeros((7, ))
        A11 = pos_wrist[0]
        A12 = -pos_wrist[2] * np.cos(q1) + pos_wrist[1] * np.sin(q1)

        A = np.array([[A11, A12],[A12, -A11]])
        b = np.array([[-0.3 * np.cos(q4) * np.sin(q3)],[0.3 - 0.3 * np.sin(q4)]])
        sol = np.dot(np.linalg.inv(A), b)
        q2 = np.arctan2(sol[1, 0], sol[0, 0])
        q_first4 = np.array([q1,q2,q3,q4]).copy()
        q_last3 = self.ik_last3(quat_wrist, q_first4)
        sol1 = np.concatenate([q_first4, q_last3]).copy()
        #the second solution
        q1 = np.pi - np.arcsin(-0.3 * np.cos(q3) * np.cos(q4)/np.sqrt(pos_wrist[1]**2 + pos_wrist[2]**2)) - np.arctan(pos_wrist[1]/pos_wrist[2])
        A11 = pos_wrist[0]
        A12 = -pos_wrist[2] * np.cos(q1) + pos_wrist[1] * np.sin(q1)
        A = np.array([[A11, A12],[A12, -A11]])
        b = np.array([[-0.3 * np.cos(q4) * np.sin(q3)],[0.3 - 0.3 * np.sin(q4)]])
        sol = np.dot(np.linalg.inv(A), b)
        q2 = np.arctan2(sol[1, 0], sol[0, 0])
        q_first4 = np.array([q1,q2,q3,q4]).copy()
        q_last3 = self.ik_last3(quat_wrist, q_first4)
        sol2 = np.concatenate([q_first4, q_last3]).copy()
        if(np.linalg.norm(q -sol1) > np.linalg.norm(q - sol2)):
            sol = sol2
        else:
            sol = sol1
        return sol

    def sol_4(self, pos_wrist,quat_wrist, q):
        '''
        generate the third solution , the third joint angle is fixed
        pos_wirst,: the position of the wirst
        quat_wrist, the quaternion of the wrist
        q1: joint angle of the first joint
        '''
        q7 = q[6]
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist)**2)/(2 * 0.3 * 0.3), -1, 1))
        #the first solition
        M = R.from_quat([quat_wrist[1], quat_wrist[2], quat_wrist[3], quat_wrist[0]])
        M = M.as_matrix()
       
        A = (-M[1,1] * M[2,0] + M[1,0] * M[2,1]) * pos_wrist[0] + (M[0,1] * M[2,0] - M[0,0] * M[2,1])*pos_wrist[1] + (-M[0,1] * M[1,0] + M[0,0] * M[1,1])*pos_wrist[2]
        B = (-M[1,2] * M[2,1] + M[1,1] * M[2,2]) * pos_wrist[0] + (M[0,2] * M[2,1] - M[0,1] * M[2,2])*pos_wrist[1] + (-M[0,2] * M[1,1] + M[0,1] * M[1,2])*pos_wrist[2]
        C = (-M[1,2] * M[2,0] + M[1,0] * M[2,2]) * pos_wrist[0] + (M[0,2] * M[2,0] - M[0,0] * M[2,2])*pos_wrist[1] + (-M[0,2] * M[1,0] + M[0,0] * M[1,2])*pos_wrist[2]
        

        righthand = -C * np.cos(q7) + B * np.sin(q7)

        #the first solution

        if(abs(righthand/np.sqrt((-0.3 * np.cos(q4))**2 + (-0.3 + 0.3 * np.sin(q4))**2))>1):
            return np.zeros((7,))
        q5 = np.arcsin(righthand/np.sqrt((-0.3 * np.cos(q4))**2 + (-0.3 + 0.3 * np.sin(q4))**2)) - np.arctan((-0.3 * np.cos(q4))/(-0.3 + 0.3 * np.sin(q4)))
        
        cosq6 = -A/(np.cos(q5) * (-0.3 + 0.3 * np.sin(q4)) + 0.3 * np.cos(q4) * np.sin(q5))
        sinq6 = (-B * np.cos(q7) - C * np.sin(q7))/(np.cos(q5) * (0.3 - 0.3 * np.sin(q4)) - 0.3 * np.cos(q4)*np.sin(q5))
        q6 = np.arctan2(sinq6, cosq6)

        q_last4 = np.array([q4,q5,q6,q7])
        q_first3 = self.ik_first3_fromlast4(q_wrist_ref=quat_wrist, q_last4 = q_last4)

        sol1 = np.concatenate([q_first3, q_last4])

        q5 = -np.arcsin(righthand/np.sqrt((-0.3 * np.cos(q4))**2 + (-0.3 + 0.3 * np.sin(q4))**2)) - np.arctan((-0.3 * np.cos(q4))/(-0.3 + 0.3 * np.sin(q4))) - np.pi
        
        cosq6 = -A/(np.cos(q5) * (-0.3 + 0.3 * np.sin(q4)) + 0.3 * np.cos(q4) * np.sin(q5))
        sinq6 = (-B * np.cos(q7) - C * np.sin(q7))/(np.cos(q5) * (0.3 - 0.3 * np.sin(q4)) - 0.3 * np.cos(q4)*np.sin(q5))
        q6 = np.arctan2(sinq6, cosq6)

        q_last4 = np.array([q4,q5,q6,q7])
        q_first3 = self.ik_first3_fromlast4(q_wrist_ref=quat_wrist, q_last4 = q_last4)

        sol2 = np.concatenate([q_first3, q_last4])

        if(np.linalg.norm(q -sol1) > np.linalg.norm(q - sol2)):
            sol = sol2
        else:
            sol = sol1
        return sol

    def sol_5(self, pos_wrist,quat_wrist, q):
        '''
        generate the third solution , the third joint angle is fixed
        pos_wirst,: the position of the wirst
        quat_wrist, the quaternion of the wrist
        q1: joint angle of the first joint
        '''
        q5 = q[4]
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist)**2)/(2 * 0.3 * 0.3), -1, 1))
        #the first solition
        M = R.from_quat([quat_wrist[1], quat_wrist[2], quat_wrist[3], quat_wrist[0]])
        M = M.as_matrix()

        A = (-M[1,1] * M[2,0] + M[1,0] * M[2,1]) * pos_wrist[0] + (M[0,1] * M[2,0] - M[0,0] * M[2,1])*pos_wrist[1] + (-M[0,1] * M[1,0] + M[0,0] * M[1,1])*pos_wrist[2]
        B = (-M[1,2] * M[2,1] + M[1,1] * M[2,2]) * pos_wrist[0] + (M[0,2] * M[2,1] - M[0,1] * M[2,2])*pos_wrist[1] + (-M[0,2] * M[1,1] + M[0,1] * M[1,2])*pos_wrist[2]
        C = (-M[1,2] * M[2,0] + M[1,0] * M[2,2]) * pos_wrist[0] + (M[0,2] * M[2,0] - M[0,0] * M[2,2])*pos_wrist[1] + (-M[0,2] * M[1,0] + M[0,0] * M[1,2])*pos_wrist[2]
        
        righthand  = -(-0.3 * np.cos(q4) * np.cos(q5) + (-0.3 + 0.3 * np.sin(q4)) * np.sin(q5))

        if(abs(righthand/np.sqrt(B**2 + C**2))>1):
            return np.zeros((7, ))

        if(B > 0):
            q7 = np.arcsin(righthand/np.sqrt(B**2 + C**2)) - np.arctan(-C/B)
            if(q7 > 0):
                q7 = -np.arcsin(righthand/np.sqrt(B**2 + C**2)) - np.arctan(-C/B) - np.pi
        else:
            q7 = np.arcsin(-righthand/np.sqrt(B**2 + C**2)) - np.arctan(-C/B)
            if(q7 > 0):
                q7 = -np.arcsin(-righthand/np.sqrt(B**2 + C**2)) - np.arctan(-C/B) - np.pi

        q6 =np.arctan((C * np.sin(q7) + B*np.cos(q7))/(-A))
        q_last4 = np.array([q4,q5,q6,q7])
        q_first3 = self.ik_first3_fromlast4(q_wrist_ref=quat_wrist, q_last4 = q_last4)

        sol = np.concatenate([q_first3, q_last4])
        return sol

    def sol_6(self, pos_wrist,quat_wrist, q):
        '''
        generate the third solution , the third joint angle is fixed
        pos_wirst,: the position of the wirst
        quat_wrist, the quaternion of the wrist
        q1: joint angle of the first joint
        '''
        q6 = q[5]
        q4 = np.pi/2 - np.arccos(np.clip((0.3**2 + 0.3 **2 - np.linalg.norm(pos_wrist)**2)/(2 * 0.3 * 0.3), -1, 1))
        M = R.from_quat([quat_wrist[1], quat_wrist[2], quat_wrist[3], quat_wrist[0]])
        M = M.as_matrix()
        A = (-M[1,1] * M[2,0] + M[1,0] * M[2,1]) * pos_wrist[0] + (M[0,1] * M[2,0] - M[0,0] * M[2,1])*pos_wrist[1] + (-M[0,1] * M[1,0] + M[0,0] * M[1,1])*pos_wrist[2]
        B = (-M[1,2] * M[2,1] + M[1,1] * M[2,2]) * pos_wrist[0] + (M[0,2] * M[2,1] - M[0,1] * M[2,2])*pos_wrist[1] + (-M[0,2] * M[1,1] + M[0,1] * M[1,2])*pos_wrist[2]
        C = (-M[1,2] * M[2,0] + M[1,0] * M[2,2]) * pos_wrist[0] + (M[0,2] * M[2,0] - M[0,0] * M[2,2])*pos_wrist[1] + (-M[0,2] * M[1,0] + M[0,0] * M[1,2])*pos_wrist[2]


        if(abs(-A * np.tan(q6)/np.sqrt(C**2 + B**2))>1):
            return np.zeros((7,))
        if(C>0):
            q7 = np.arcsin(-A * np.tan(q6)/np.sqrt(C**2 + B**2)) - np.arctan(B/C)
            if(q7>0):
                q7 = -np.arcsin(-A * np.tan(q6)/np.sqrt(C**2 + B**2)) - np.arctan(B/C) -np.pi
        else:
            q7 = np.arcsin(A * np.tan(q6)/np.sqrt(C**2 + B**2)) - np.arctan(B/C)
            if(q7>0):
                q7 = -np.arcsin(A * np.tan(q6)/np.sqrt(C**2 + B**2)) - np.arctan(B/C) -np.pi
        
        Q = np.zeros((2,2))
        Q[0,0] = -0.3 * np.cos(q4)
        Q[1,1] = 0.3 * np.cos(q4)
        Q[0,1] = -0.3 + 0.3 * np.sin(q4)
        Q[1,0] = -0.3 + 0.3 * np.sin(q4)

        b = np.zeros((2,1))
        b[0, 0] = C * np.cos(q7) - B * np.sin(q7)
        b[1, 0] = -A * np.cos(q6) + np.sin(q6) * (B * np.cos(q7) + C * np.sin(q7))
        sol = np.dot(np.linalg.inv(Q), b)

        q5 = np.arctan2(sol[1, 0], sol[0, 0])

        q_last4 = np.array([q4,q5,q6,q7])
        q_first3 = self.ik_first3_fromlast4(q_wrist_ref=quat_wrist, q_last4 = q_last4)
        q = np.concatenate([q_first3, q_last4])
        return q

    

    def ik(self, transformation, q_init = False):
        '''
        given the target transformation of the chopsticks, compute the joint angles of the arm
        q_init: the arm pose in the last frame
        '''
        pos_wrist_ref, q_wrist_ref = self.compute_target_transformation(transformation)
        pos_elbow_approx, pos_wrist_approx = self.linear_model(pos_wrist_ref)
        q_first4 = self.ik_first4(pos_elbow_approx, pos_wrist_approx)
        q_last3 = self.ik_last3(q_wrist_ref, q_first4)
        q =  np.concatenate([q_first4, q_last3])
       
        def f(x):
            self.sim.data.qpos[0:7] = x
            self.sim.forward()
            pos_chopstick2 = self.sim.data.body_xpos[self.sim.model.body_name2id('chopstick2')].copy()
            q_chopstick2 = self.sim.data.body_xquat[self.sim.model.body_name2id('chopstick2')].copy()
            dq = glm.inverse(convert_quat2glm(transformation[3:])) * convert_quat2glm(q_chopstick2) 
            if(dq[3]<0):
                dq= -dq
            loss = 0.2 * np.arccos(np.clip(dq[3], -1, 1))
            loss += np.linalg.norm(pos_chopstick2 - transformation[0:3])
            return loss
        bound = self.sim.model.jnt_range[0:7,:]
        bound = list(zip(*(bound.T)))
        if(type(q_init) == bool):
            print('init planning')
            init_pos = q
            fun = 100
            while(fun >1e-4):
                res = minimize(f, init_pos + np.random.uniform([-0.5] * 7,[0.5]*7), bounds= bound, method="L-BFGS-B")
                fun = res.fun
                print(fun)
        else:
            init_pos = q_init
            fun = 100
            if(f(init_pos)>0.1):
                embed()
            while(fun >1e-4):
                res = minimize(f, init_pos + np.random.uniform([-0.01] * 7,[0.01]*7), bounds= bound, method="L-BFGS-B", options = {'maxiter': 50, 'disp': False, 'maxls':10})
                fun = res.fun
                print(fun)
        q = res.x
        return q



    def check_bound(self, x):
        return (x>self.jnt_range[:,0]).all() and (x<self.jnt_range[:,1]).all()

    
    def ik_seq(self, motions_chopsticks, q_init = False):
        '''
        given sequences of motions of the chopsticks, output corresponding sequences
        motions: (N, 7) the motions of the chopsticks
        ouuput: (N, 7)
        '''
        start = time.time()
        if(math.isnan(motions_chopsticks[0,-1])):
            embed()
        q = q_init
        motions_arm = np.zeros((motions_chopsticks.shape[0], 7 + 1))
        motions_arm[:-1,0] = 0.01
        motions_arm[0, 1:] = q.copy()
        for i in range(1, motions_arm.shape[0]):
            pose_chopsticks = motions_chopsticks[i,1:]
            pos_wrist, q_wrist = self.compute_target_transformation(pose_chopsticks)
            pos_wrist = pos_wrist - self.pos_shoulder
            pose_arm_list = []
            sol1 = self.sol_1(pos_wrist, q_wrist, motions_arm[i-1,1:])
            if(self.check_bound(sol1)):
                pose_arm_list += [sol1]

            sol2 = self.sol_2(pos_wrist, q_wrist, motions_arm[i-1,1:])
            if(self.check_bound(sol2)):
              pose_arm_list += [sol2]
            
            sol3 = self.sol_3(pos_wrist, q_wrist, motions_arm[i-1,1:])
            if(self.check_bound(sol3)):
               pose_arm_list += [sol3]

            sol4 = self.sol_4(pos_wrist, q_wrist, motions_arm[i-1,1:])
            if(self.check_bound(sol4)):
               pose_arm_list += [sol4]

            sol5 = self.sol_5(pos_wrist, q_wrist, motions_arm[i-1,1:])
            if(self.check_bound(sol5)):
                pose_arm_list += [sol5]

            sol6 = self.sol_6(pos_wrist, q_wrist, motions_arm[i-1,1:])
            if(self.check_bound(sol6)):
               pose_arm_list += [sol6]
            if(len(pose_arm_list)!=0):
                pose_arm_array = np.array(pose_arm_list)
                jnt_lower_bound = self.jnt_range[:,0].reshape((1,-1))
                jnt_upper_bound = self.jnt_range[:,1].reshape((1,-1))
                error_continuous = ((pose_arm_array - motions_arm[i-1,1:])**2).sum(1)
                error_natural = ((pose_arm_array[:,4:6])**2).sum(1)
                error = error_continuous + 0.1 * error_natural
                idx = error.argsort()[0]
                pose_arm = pose_arm_list[idx].copy()
            else:
                print('no feasiable solution')
                return False   
            motions_arm[i,1:] = pose_arm
        end = time.time()
        print("dt ik:{}".format(end - start))
        #debug
        if(np.linalg.norm(motions_arm[-2,1] - motions_arm[-1,1])> 0.2):
            embed()
        return motions_arm

    def solve_pos(self, object_idx, pos_object, dr, dq, init_config):
        '''
        given the position of the object and initial configuration of the hand, compute the target position and transformation of the object
        '''
        object_name = 'object_' + str(object_idx)
        geom_object = self.sim.model.geom_name2id(object_name)
        geom_palm = self.sim.model.geom_name2id('palm_virtual')

        def f(x):
            self.sim.data.qpos[0:7] = x
            self.sim.forward()
            pos_chopstick2 = self.sim.data.body_xpos[self.sim.model.body_name2id('chopstick2')].copy()
            q_chopstick2 = self.sim.data.body_xquat[self.sim.model.body_name2id('chopstick2')].copy()
            loss = 0
            for j in range(self.sim.data.ncon):
                    contact = self.sim.data.contact[j]
                    if((contact.geom2 in self.geom_chopsticks and contact.geom1 != geom_object)):
                        dis_contact = contact.dist
                        loss += np.min([np.exp(1e4 * abs(dis_contact)), 1e6]) - 1
                    if((contact.geom1 in self.geom_chopsticks and contact.geom2 != geom_object)):
                        dis_contact = contact.dist
                        loss += np.min([np.exp(1e4 * abs(dis_contact)), 1e6]) - 1
                    if((contact.geom2 not in self.geom_chopsticks and contact.geom1 == geom_object)):
                        dis_contact = contact.dist
                        loss += np.exp(10000 * abs(dis_contact)) - 1
                    if((contact.geom1 not in self.geom_chopsticks and contact.geom2 == geom_object)):
                        dis_contact = contact.dist
                        loss += np.exp(10000 * abs(dis_contact)) - 1
                    if((contact.geom1 == geom_palm)):
                        dis_contact = contact.dist
                        loss += np.exp(10000 * abs(dis_contact)) - 1
                    if((contact.geom2 == geom_palm)):
                        dis_contact = contact.dist
                        loss += np.exp(10000 * abs(dis_contact)) - 1

            transform_object =  compute_transform_object_from_chopsticks(np.concatenate([pos_chopstick2, q_chopstick2]), dr, dq)
            loss = np.linalg.norm(transform_object[0:3] - pos_object)
            return loss

        bound = self.sim.model.jnt_range[0:7,:]
        bound[4,:] = np.array([-0.3, 0.3])
        bound[5,:] = np.array([-0.3, 0.3])
        bound = list(zip(*(bound.T)))
      
        init_pos = init_config
        fun = 100
        while(fun >1e-4):
            res = minimize(f, init_pos + np.random.uniform([-0.5] * 7,[0.5]*7), bounds= bound, method="L-BFGS-B")
            fun = res.fun
            x = res.x
        self.sim.data.qpos[0:7] = x
        self.sim.forward()
        pos_chopstick2 = self.sim.data.body_xpos[self.sim.model.body_name2id('chopstick2')].copy()
        q_chopstick2 = self.sim.data.body_xquat[self.sim.model.body_name2id('chopstick2')].copy()
        transform_object =  compute_transform_object_from_chopsticks(np.concatenate([pos_chopstick2, q_chopstick2]), dr, dq)
        return transform_object
        