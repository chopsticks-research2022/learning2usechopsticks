import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurationNet(nn.Module):
    '''
    the class of the configuration neural network which is to find the grasping configuration of 
    an object
    '''

    def __init__(self, input_dim, output_dim, hidden=[512, 512]):
        super().__init__()
        self.fc = []
        input_dim = input_dim
        for h_dim in hidden:
            self.fc.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        self.fc = nn.ModuleList(self.fc)
        self.fca = nn.Linear(input_dim, output_dim)
        self.activation = F.tanh

    def forward(self, x):
        # normalize x first
        layer = x
        for fc_op in self.fc:
            layer = self.activation(fc_op(layer))
        # unnormalize action
        layer_a = self.fca(layer)
        output = layer_a
        return output


class ReachNet(nn.Module):
  '''
  the class of reachability neural network which is used to predict whether an object is reachable 
  given a configuration of the chopsticks 
  '''

  def __init__(self, input_dim, output_dim, hidden=[256, 256]):
    super().__init__()
    self.fc = []
    input_dim = input_dim
    for h_dim in hidden:
      self.fc.append(nn.Linear(input_dim, h_dim))
      input_dim = h_dim
    self.fc = nn.ModuleList(self.fc)
    self.fca =nn.Linear(input_dim, output_dim)
    self.activation = F.tanh

  def forward(self, x):
    # normalize x first
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))
    # unnormalize action
    layer_a = self.fca(layer)
    output = torch.sigmoid(layer_a)
    return output



def train_configuration_network(input_path, label_path, num_epoch=20, batch_size=512, lr=1e-3, mom=0.9):
    from model.dataset import ConfigurationDataset
    training_data = ConfigurationDataset(input_path, label_path)
    model = ConfigurationNet(4, 2000, [512, 512])
    train_network('configuration', model, training_data, num_epoch, batch_size, lr, mom)


def train_reach_network(path = None, num_epoch=100, batch_size=256, lr=1e-3, mom=0.9):
    from model.dataset import ReachDataset
    training_data = ReachDataset(path)
    model = ReachNet(9, 1, [256, 256])
    train_network('reach', model, training_data, num_epoch, batch_size, lr, mom)


def test_reach_net(model_path, data_path=None):
    from torch.utils.data import DataLoader
    from model.dataset import ReachDataset
    
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_data = ReachDataset(data_path)
    train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)
    model = ReachNet(9, 1, [256, 256])
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    num_samples = 0
    num_true = 0
    with torch.no_grad():
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            num_samples += 512
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.float().to(device).view((-1,1)).int()
            outputs = model(inputs)
            outputs = (outputs>0.5).int()
            num_true += (outputs == labels).sum()
    print('accuracy:{}'.format(num_true/num_samples))


def train_network(network_name, model, training_data, num_epoch=20, batch_size=512, lr=1e-3, mom=0.9):
    from torch.utils.data import DataLoader

    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum= mom)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        num_samples = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            num_samples += batch_size
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize\
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('epoch:{}'.format(epoch))
        print("average BCE loss:{}".format(running_loss/num_samples))
           
    model.cpu()
    torch.save(model.state_dict(), f'./data/{network_name}/{network_name}net_mlp.pth')
    print('Finished Training')