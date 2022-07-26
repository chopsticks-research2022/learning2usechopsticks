import numpy as np
import glm

def convert_quat2glm(x):
    '''
    takes in a 4 elements array as quaternion (w, x, y, z)
    and converts it into glm quat
    '''
    return glm.quat(x[0],x[1],x[2],x[3])

def convert_glm2quat(x):
    '''
    takes in a glm quat and converts it into an array
    representing a quaternion in (w, x, y, z)
    '''
    return np.array([x[3],x[0],x[1],x[2]])

def convert_pos2glm(x):
    '''
    takes in a 3 elements array as position (x, y, z)
    and converts it into glm quat
    '''
    return glm.quat(0, x[0],x[1],x[2])

def convert_glm2pos(x):
    '''
    takes in a glm quat and converts it into an array
    representing a position value in (x, y, z)
    '''
    return np.array([x[0],x[1],x[2]])
