import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from IPython import embed
######
INIT_ACTOR_SCALE = 0.01
NOISE = 0.1
USE_ELU = False

NORM_SAMPLES    = 1e8
######

class Normalizer(nn.Module):
  '''
    the class of state normalization modeule
  '''
  def __init__(self, in_dim, dim_ignore = 0, sample_lim=NORM_SAMPLES):
    '''
    dim ignore: the how many last dimensions of the features are not taken into considerations  
    '''
    super(Normalizer, self).__init__()

    self.mean    = nn.Parameter(torch.zeros([in_dim]))
    self.std     = nn.Parameter(torch.ones([in_dim]))
    self.mean_sq = nn.Parameter(torch.ones([in_dim]))
    self.num     = nn.Parameter(torch.zeros([1]))
    self.dim_ignore = dim_ignore
    self.in_dim = in_dim
    
    self.sum_new    = torch.zeros([in_dim])
    self.sum_sq_new = torch.zeros([in_dim])
    self.num_new    = torch.zeros([1])

    for param in self.parameters():
      param.requires_grad = False

    self.sample_lim = sample_lim


  def forward(self, x):
    return (x - self.mean) / self.std

  def unnormalize(self, x):
    return x * self.std + self.mean

  def set_mean_std(self, mean, std):
    self.mean.data = torch.Tensor(mean)
    self.std.data = torch.Tensor(std)

  def record(self, x):
    if (self.num + self.num_new >= self.sample_lim):
      return
    if x.dim() == 1:
      self.num_new += 1
      self.sum_new += x
      self.sum_new[self.in_dim -self.dim_ignore:] = 0
      self.sum_sq_new += torch.pow(x, 2)
      self.sum_sq_new[self.in_dim - self.dim_ignore:] = 0
    elif x.dim() == 2:
      self.num_new += x.shape[0]
      self.sum_new += torch.sum(x, dim=0)
      self.sum_new[self.in_dim-self.dim_ignore:] = 0
      self.sum_sq_new += torch.sum(torch.pow(x, 2), dim=0)
      self.sum_sq_new[self.in_dim - self.dim_ignore: ] = 0
    elif(x.dim() == 3):
      x = x.view((-1, x.shape[-1]))
      self.num_new += x.shape[0]
      self.sum_new += torch.sum(x, dim=0)
      self.sum_new[self.in_dim-self.dim_ignore:] = 0
      self.sum_sq_new += torch.sum(torch.pow(x, 2), dim=0)
      self.sum_sq_new[self.in_dim - self.dim_ignore: ] = 0
    else:
      assert(False and "normalizer record more than 2 dim")

  def update(self):
    if self.num >= self.sample_lim or self.num_new == 0:
      return
    # update mean, mean_sq and std
    total_num = self.num + self.num_new;
    self.mean.data *= (self.num / total_num)
    self.mean.data += self.sum_new / total_num
    self.mean_sq.data *= (self.num / total_num)
    self.mean_sq.data += self.sum_sq_new / total_num
    self.std.data = torch.sqrt(torch.abs(self.mean_sq.data - torch.pow(self.mean.data, 2)))
    self.std.data += 0.01 # in case of divide by 0
    self.std.data[self.in_dim - self.dim_ignore:] = 1
    self.num.data += self.num_new
    # clear buffer
    self.sum_new.data.zero_()
    self.sum_sq_new.data.zero_()
    self.num_new.data.zero_()

# initialize fc layer using xavier uniform initialization
def xavier_init(module):
  nn.init.xavier_uniform_(module.weight.data, gain=1)
  nn.init.constant_(module.bias.data, 0)
  return module

def orthogonal_init(module, gain=1):
  nn.init.orthogonal_(module.weight.data, gain)
  nn.init.constant_(module.bias.data, 0)
  return module

class Actor(nn.Module):
  '''
  the class of actor neural network
  '''

  def __init__(self, s_dim, a_dim, a_bound, hidden=[1024, 512], noise = 0.1, noise_mask = None, mode = 'sample'):
    '''
    s_dim: state dimentions
    a_dim: action dimentions
    a_bound: action bounds
    hidden : dimentions of differnet layers
    '''
    super(Actor, self).__init__()
    self.s_norm = Normalizer(s_dim)
    self.mode = mode
    self.fc = []
    input_dim = s_dim
    for h_dim in hidden:
      self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
      input_dim = h_dim

    self.fc = nn.ModuleList(self.fc)

    # initialize final layer weight to be [INIT, INIT]
    # 
    self.fca =orthogonal_init(nn.Linear(input_dim, a_dim),0.001)
    # nn.init.uniform_(self.fca.weight, -0.001, 0.001)
    # nn.init.constant_(self.fca.bias, 0)

    # set a_norm not trainable
    self.a_min = nn.Parameter(torch.Tensor(a_bound.low))
    self.a_max = nn.Parameter(torch.Tensor(a_bound.high))
    self.a_mean = nn.Parameter((self.a_max + self.a_min) / 2)
    scale = (a_bound.high -a_bound.low)/2
    self.scale = nn.Parameter(torch.tensor(scale).float())
    a_std = np.array([noise]*a_bound.shape[0])
    if(np.sum(noise_mask)!=None):
      a_std = a_std * noise_mask
    print("a_std:{}".format(a_std))
    a_std = np.log(a_std)
    self.a_std  = nn.Parameter(torch.Tensor(a_std).float())
    
    self.a_min.requires_grad = False
    self.a_max.requires_grad = False
    self.a_mean.requires_grad = False
    self.a_std.requires_grad = True
    self.scale.requires_grad = False
    #print(self.scale)

    self.activation = torch.tanh

  def fix_std(self):
    #fix the action noise
    self.a_std.requires_grad = False


  def set_std(self, std):
    self.a_std.data = torch.Tensor(np.log(std))
    print('reset std:{}'.format(self.a_std.data))

  def forward(self, x):
    # normalize x first
    x = self.s_norm(x)
    if(self.mode == 'sample' and  (not self.is_fixed())):
      #print('record')
      self.s_norm.record(x)
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))

    # unnormalize action
    layer_a = self.fca(layer)
    #print(layer_a)
    a_mean = layer_a * self.scale
    #print(a_mean)
   # a_mean = self.activation(a_mean)

    return a_mean
  def set_mode(self, mode):
    if mode not in ['sample', 'deter']:
      raise NotImplementedError
    self.mode = mode

  def update(self):
    self.s_norm.update()

  def act_distribution(self, x):
    '''
    compute action distributions
    '''
    a_mean = self.forward(x)
    m = D.Normal(a_mean, torch.exp(self.a_std))
    return m

  def act_deterministic(self, x):
    '''
    compute determnistic actions
    '''
    return self.forward(x)

  def act_stochastic(self, x):
    m = self.act_distribution(x)
    ac = m.sample()
    return ac

  def fix_parameters(self):
    #fix the parameters of the model
    for param in self.parameters():
      param.requires_grad = False

  def is_fixed(self):
    flag = True
    for param in self.parameters():
      flag = flag and not (param.requires_grad)
    return flag

  def train_parameters(self):
    for name, param in self.named_parameters():
        if name not in ['a_min', 'a_max', 'a_mean', 'scale', 'a_std', 's_norm.mean', 's_norm.std', 's_norm.mean_sq', 's_norm.num']: 
          param.requires_grad = True
    #for name, param in self.named_parameters():
    #    print(name)
    #    print(param.data)
    #    print(param.requires_grad)



class Gating(nn.Module):
  '''
  the class of the gating network
  '''
  def __init__(self,s_dim, gating_dim, hidden_gating, mode = 'sample'):
      super(Gating, self).__init__()
      self.fc = []
      self.mode = mode
      #gating network
      input_dim = s_dim
      self.s_norm = Normalizer(s_dim)
      for h_dim in hidden_gating:
        self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
        input_dim = h_dim

      self.fc = nn.ModuleList(self.fc)
      # initialize final layer weight to be [INIT, INIT]
      self.fca =orthogonal_init(nn.Linear(input_dim, gating_dim))
      self.activation = torch.tanh

  def forward(self, x):
      x_norm = self.s_norm(x)
      if(self.mode == 'sample'):
        self.s_norm.record(x)
      layer = x_norm
      for fc_op in self.fc:
        layer = self.activation(fc_op(layer))

      # unnormalize action
      layer_a = self.fca(layer)
      #print(layer_a)
      gating_score = layer_a
      return gating_score


  def set_mode(self, mode):
      self.mode = mode

  def update(self):
      self.s_norm.update()
    








class MOE(nn.Module):
  '''
  the class of Mixture of Expert network:consisting of a gating network and 4 expert policy network
  '''

  def __init__(self, s_dim, gating_dim, a_dim, a_bound, hidden_gating=[1024, 512], hidden_expert = [1024, 512], noise = 0.1,  noise_mask = None, mode = 'sample'):
    '''
    s_dim: state dimentions
    a_dim: action dimentions
    a_bound: action bounds
    hidden : dimentions of differnet layers
    '''
    super(MOE, self).__init__()
    self.mode = mode
    self.gating = Gating(s_dim, gating_dim, hidden_gating, mode)
    #gating network
    self.expert1 = Actor(s_dim - 20, a_dim, a_bound, hidden = hidden_expert)
    self.expert2 = Actor(s_dim - 20, a_dim, a_bound, hidden = hidden_expert)
    self.expert3 = Actor(s_dim - 20, a_dim, a_bound, hidden = hidden_expert)
    self.expert4 = Actor(s_dim - 20, a_dim, a_bound, hidden = hidden_expert)
    #fix the action noise of the expert policy
    self.expert1.fix_std()
    self.expert2.fix_std()
    self.expert3.fix_std()
    self.expert4.fix_std()

    a_std = np.array([noise]*a_bound.shape[0])
    if(np.sum(noise_mask)!=None):
      a_std = a_std * noise_mask
    #print("a_std:{}".format(a_std))
    a_std = np.log(a_std)
    self.a_std  = nn.Parameter(torch.Tensor(a_std).float())
    self.a_std.requires_grad = True

  def set_mode(self, mode):
    self.mode = mode
    self.gating.set_mode(mode)
    self.expert1.set_mode(mode)
    self.expert2.set_mode(mode)
    self.expert3.set_mode(mode)
    self.expert4.set_mode(mode)

  def update(self):
      self.gating.update()
      self.expert1.update()
      self.expert2.update()
      self.expert3.update()
      self.expert4.update()

  def set_std(self, std):
    self.a_std.data = torch.Tensor(np.log(std))
    print('reset std:{}'.format(self.a_std.data))

  def forward(self, x):
    gating_score = F.softmax(self.gating(x), -1).unsqueeze(-2) #(batch_size, 1, 4) or (1, 4)
    #print(gating_score)
    #gating_score = torch.Tensor(np.array([[0,1,0,0]]))
    if(len(x.shape) ==2):
      x_sub = x[:,0:-20]
    else:
      x_sub = x[0:-20]
    
    action1 = self.expert1(x_sub).unsqueeze(-1) #(batch_size, a_dim,1) or (a_dim,1)
    action2 = self.expert2(x_sub).unsqueeze(-1) #(batch_size, a_dim,1) or (a_dim,1)
    action3 = self.expert3(x_sub).unsqueeze(-1) #(batch_size, a_dim, 1) or (a_dim,1)
    action4 = self.expert4(x_sub).unsqueeze(-1) #(batch_size, a_dim,1) or (a_dim,1)

    a_mean = torch.cat([action1, action2, action3, action4], -1) #(batch_size, a_dim, 4) or (a_dim, 4)
    a_mean = (a_mean * gating_score).sum(-1)
    #print(a_mean)
   # a_mean = self.activation(a_mean)

    return a_mean

  def act_distribution(self, x):
    '''
    compute action distributions
    '''
    a_mean = self.forward(x)
    m = D.Normal(a_mean, torch.exp(self.a_std))
    return m

  def load_experts(self, model_path):
    #load expert networks from path
    model_pregrasp = model_path[0]
    model_lift = model_path[1]
    model_move = model_path[2]
    model_release = model_path[3]

    _, self.expert1, _ = load_model(model_pregrasp)
    _, self.expert2, _ = load_model(model_lift)
    _, self.expert3, _ = load_model(model_move)
    _, self.expert4, _ = load_model(model_release)

    for name, param in self.named_parameters():
        print(name)
        print(param.data)
        print(param.requires_grad)

  def load_gating(self,gating_path):
    self.gating = load_gating(gating_path)

  def fix_experts(self):
    self.expert1.fix_parameters()
    self.expert2.fix_parameters()
    self.expert3.fix_parameters()
    self.expert4.fix_parameters()

  def train_experts(self):
    self.expert1.train_parameters()
    self.expert2.train_parameters()
    self.expert3.train_parameters()
    self.expert4.train_parameters()

  def act_deterministic(self, x):
    '''
    compute determnistic actions
    '''
    return self.forward(x)

  def act_stochastic(self, x):
    m = self.act_distribution(x)
    ac = m.sample()
    return ac





class Critic(nn.Module):
      
  '''
  the class of critic network
  '''

  def __init__(self, s_dim, val_min, val_max, hidden=[1024, 512], mode = 'sample'):
    super(Critic, self).__init__()
    self.fc = []
    self.mode = mode
    self.s_norm = Normalizer(s_dim)
    input_dim = s_dim
    for h_dim in hidden:
      self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
      input_dim = h_dim

    self.fc = nn.ModuleList(self.fc)
    self.fcv = orthogonal_init(nn.Linear(input_dim, 2))
    self.v_min = torch.Tensor([val_min])
    self.v_max = torch.Tensor([val_max])
    self.v_mean = nn.Parameter((self.v_max + self.v_min) / 2)
    self.v_std  = nn.Parameter((self.v_max - self.v_min) / 2)
    self.v_min.requires_grad = False
    self.v_max.requires_grad = False
    self.v_mean.requires_grad = False
    self.v_std.requires_grad = False
    self.activation = F.relu

  def forward(self, x):
    '''
    compute state values
    '''
    x = self.s_norm(x)
    if(self.mode =='sample'):
      self.s_norm.record(x)
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))
    # unnormalize value
    value = self.fcv(layer)
    value = self.v_std * value + self.v_mean
    return value

  def set_mode(self, mode):
    if mode not in ['sample', 'deter']:
      raise NotImplementedError
    self.mode = mode

  def update(self):
    self.s_norm.update()

def load_model(ckpt, mode = 'single'):
  '''
  load the saved model

  ckpt : path of loaded model
  return :
        s_norm: state normalization module
        actor: actor neural network
        critic : critic neural network
  '''
  if(mode == 'single'):
    data = torch.load(ckpt)
    s_dim = data["s_norm"]["mean"].shape[0]
    a_dim = data["actor"]["fca.bias"].shape[0]
    a_min = data["actor"]["a_min"].numpy()
    a_max = data["actor"]["a_max"].numpy()
    import gym.spaces
    a_bound = gym.spaces.Box(a_min, a_max, dtype=np.float32)
    a_hidden = list(map(lambda i: data["actor"]["fc.%d.bias" % i].shape[0], [0, 1]))
    c_hidden = list(map(lambda i: data["critic"]["fc.%d.bias" % i].shape[0], [0, 1]))
    s_norm = Normalizer(s_dim, sample_lim=-1)
    actor = Actor(s_dim, a_dim, a_bound, a_hidden)
    critic = Critic(s_dim, 0, 1, c_hidden)
    s_norm.load_state_dict(data["s_norm"])
    actor.load_state_dict(data["actor"])
    critic.load_state_dict(data["critic"])
    if("qpos_buffer" in data.keys()):
      qpos_buffer = data["qpos_buffer"]
      qvel_buffer = data["qvel_buffer"]
      return s_norm, actor, critic, qpos_buffer, qvel_buffer
    else:
      return s_norm, actor, critic
  elif(mode == 'MOE'):
    data = torch.load(ckpt)
    s_dim = data["s_norm"]["mean"].shape[0]
    a_dim = data["actor"]["expert1.fca.bias"].shape[0]
    a_min = data["actor"]["expert1.a_min"].numpy()
    a_max = data["actor"]["expert1.a_max"].numpy()
    import gym.spaces
    a_bound = gym.spaces.Box(a_min, a_max, dtype=np.float32)
    hidden_gating = list(map(lambda i: data["actor"]["gating.fc.%d.bias" % i].shape[0], [0, 1]))
    hidden_expert = list(map(lambda i: data["actor"]["expert1.fc.%d.bias" % i].shape[0], [0, 1]))
    c_hidden = list(map(lambda i: data["critic"]["fc.%d.bias" % i].shape[0], [0, 1]))
    s_norm = Normalizer(s_dim, sample_lim=-1)
    actor = MOE(s_dim, 4, a_dim, a_bound, hidden_gating, hidden_expert)
    critic = Critic(s_dim, 0, 1, c_hidden)
    s_norm.load_state_dict(data["s_norm"])
    actor.load_state_dict(data["actor"])
    critic.load_state_dict(data["critic"])
    if("qpos_buffer" in data.keys()):
      qpos_buffer = data["qpos_buffer"]
      qvel_buffer = data["qvel_buffer"]
      return s_norm, actor, critic, qpos_buffer, qvel_buffer
    else:
      return s_norm, actor, critic
  else:
    NotImplementedError





def load_gating(ckpt):
  '''
  load the saved model of the gating network

  ckpt : path of loaded model
  return :
        s_norm: state normalization module
        actor: actor neural network
        critic : critic neural network
  '''
  data = torch.load(ckpt)
  s_dim = data["gating"]["s_norm.mean"].shape[0]
  hidden_gating = list(map(lambda i: data["gating"]["fc.%d.bias" % i].shape[0], [0, 1]))
  gating = Gating(s_dim, 4, hidden_gating)
  gating.load_state_dict(data["gating"])
  return gating
 







# ####################################
# #models used for SAC algorithm
# from rlkit.torch.sac.policies import TanhGaussianPolicy
# # from rlkit.torch.sac.sac import SoftActorCritic
# from rlkit.torch.networks import FlattenMlp
# import rlkit.torch.pytorch_util as ptu
# import os

# def create_networks(env, config):
#     """ Creates all networks necessary for SAC.

#     These networks have to be created before instantiating this class and
#     used in the constructor.

#     TODO: Maybe this should be reworked one day...

#     Args:
#         config: A configuration dictonary of q value and policy network.

#     Returns:
#         A dictonary which contains the networks.
#     """
#     obs_dim = int(np.prod(env.observation_space.shape))
#     action_dim = int(np.prod(env.action_space.shape))
#     net_size = config['net_size']
#     hidden_sizes = [net_size] * config['network_depth']
#     # hidden_sizes = [net_size, net_size, net_size]
#     qf1 = FlattenMlp(
#         hidden_sizes=hidden_sizes,
#         input_size=obs_dim + action_dim,
#         output_size=1,
#     ).to(device=ptu.device)
#     qf2 = FlattenMlp(
#         hidden_sizes=hidden_sizes,
#         input_size=obs_dim + action_dim,
#         output_size=1,
#     ).to(device=ptu.device)
#     qf1_target = FlattenMlp(
#         hidden_sizes=hidden_sizes,
#         input_size=obs_dim + action_dim,
#         output_size=1,
#     ).to(device=ptu.device)
#     qf2_target = FlattenMlp(
#         hidden_sizes=hidden_sizes,
#         input_size=obs_dim + action_dim,
#         output_size=1,
#     ).to(device=ptu.device)
#     policy = TanhGaussianPolicy(
#         hidden_sizes=hidden_sizes,
#         obs_dim=obs_dim,
#         action_dim=action_dim,
#     ).to(device=ptu.device)

#     clip_value = 1.0
#     for p in qf1.parameters():
#         p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
#     for p in qf2.parameters():
#         p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
#     for p in policy.parameters():
#         p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

#     return {'qf1' : qf1, 'qf2' : qf2, 'qf1_target' : qf1_target, 'qf2_target' : qf2_target, 'policy' : policy}


# def save_networks(networks, file_path, iteration, from_scratch):
#     """ Saves the networks on the disk.
#       file_path: path of dir where models are saved
#       iteration: id of saved models
#     """
#     checkpoints = {}
#     for key, net in networks.items():
#         checkpoints[key] = net.state_dict()
#     #file_path = os.path.join(self._config['data_folder_experiment'], 'checkpoints')
#     if not os.path.exists(file_path):
#       os.makedirs(file_path)
#     if(from_scratch):
#       torch.save(checkpoints, os.path.join(file_path, 'checkpoint_design_{}.chk'.format(iteration)))
#     else:
#       torch.save(checkpoints, os.path.join(file_path, 'transfer_checkpoint_design_{}.chk'.format(iteration)))
# def load_networks(path, networks):
#     """ Loads networks from the disk.
#     """
#     model_data = torch.load(path) #, map_location=ptu.device)
#     for key, net in networks.items():
#         params = model_data[key]
#         net.load_state_dict(params)