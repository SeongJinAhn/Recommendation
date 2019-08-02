import torch
import torch.nn as nn
from engine import Engine
from utils import use_cuda


class DMF(torch.nn.Module):
    def __init__(self, config):
        super(DMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.matrix = config['matrix']

        self.user_layers = nn.ModuleList()
        for i,in_size in enumerate(config['user_layers'][:-1]):
            out_size = config['user_layers'][i+1]
            self.user_layers.append(nn.Linear(in_size,out_size))

        self.item_layers = nn.ModuleList()
        for i,in_size in enumerate(config['item_layers'][:-1]):
            out_size = config['item_layers'][i+1]
            self.item_layers.append(nn.Linear(in_size,out_size))
           
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_vector = self.matrix[user_indices-1,:]
        item_vector = self.matrix[:,item_indices-1]
        item_vector = torch.t(item_vector)

        for i in range(len(self.user_layers)):
            user_vector = self.user_layers[i](user_vector)
            user_vector = nn.ReLU()(user_vector)

        for i in range(len(self.item_layers)):
            item_vector = self.item_layers[i](item_vector)
            item_vector = nn.ReLU()(item_vector)

        y_ = torch.mul(user_vector, item_vector).sum(dim=1)
        y_ = y_ / (user_vector.norm() * item_vector.norm())

#        y_ = self.logistic(y_)
        return y_

    def init_weight(self):
        pass


class DMFEngine(Engine):
    """Engine for training & evaluating DMF model"""
    def __init__(self, config):
        self.model = DMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(DMFEngine, self).__init__(config)
