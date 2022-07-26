import json
import numpy as np


class StatePool(object):
    def __init__(self, num_phase = 4, sample_lim_phase = 1e2) -> None:
        super().__init__()
        self.num_phase = num_phase
        self.sample_lim_phase = sample_lim_phase
        #initialize the state pool
        self.pools = []
        self.pool_init = []
        for i in range(self.num_phase):
            self.pools.append([])

    def update(self, state, phase = 0):
        self.pools[phase].append(state)
        if(len(self.pools[phase]) > self.sample_lim_phase):
            self.pools[phase].pop(0)

    def save_init(self, state):
        #save the initial kinematic state
        for i in range(1):
            self.pool_init.append(state)

    def sample(self, phase):
        idx = np.random.randint(0, len(self.pools[phase]))
        return self.pools[phase][idx]

    def len_phase(self,phase):
        return len(self.pools[phase])

    def save(self,path):
        #save the data in the state buffer
        import copy
        def process(x):
            for i in range(len(x)):
                for key in x[i].keys():
                    if(key in ['qpos', 'qvel', 'ctrl', 'qacc']):
                        x[i][key] = x[i][key].tolist()
            return x
        path_phase0 = path + 'state_0.txt'
        with open(path_phase0, 'w') as fout:
            json.dump(process(copy.deepcopy(self.pools[0])), fout)
        path_phase1 = path + 'state_1.txt'
        with open(path_phase1, 'w') as fout:
            json.dump(process(copy.deepcopy(self.pools[1])), fout)
        path_phase2 = path + 'state_2.txt'
        with open(path_phase2, 'w') as fout:
            json.dump(process(copy.deepcopy(self.pools[2])), fout)
        path_phase3 = path + 'state_3.txt'
        with open(path_phase3, 'w') as fout:
            json.dump(process(copy.deepcopy(self.pools[3])), fout)

    def load(self,path, phase):
        '''
        load the saved state list
        '''
        def process(x):
            for i in range(len(x)):
                for key in x[i].keys():
                    if(key in ['qpos', 'qvel', 'ctrl', 'qacc']):
                        x[i][key] = np.array(x[i][key])
            return x
        with open(path, 'r') as f:
            state_dict = json.load(f)
            state_dict= process(state_dict)
            self.pools[phase] = state_dict
