'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super(AGREE, self).__init__()
        self.userembeds = nn.Embedding(num_users,embedding_dim)
        self.itemembeds = nn.Embedding(num_items,embedding_dim)
        self.groupembeds = nn.Embedding(num_groups,embedding_dim)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal(m.weight)

    def forward(self, group_inputs, user_inputs, item_inputs):
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs):
        group_embeds, item_embeds = self.groupembeds(torch.LongTensor(group_inputs)), self.itemembeds(torch.LongTensor(item_inputs))
        element_embeds = torch.mul(group_embeds,item_embeds)
        new_embeds = torch.cat((element_embeds, group_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        user_embeds, item_embeds = self.userembeds(torch.LongTensor(user_inputs)), self.itemembeds(torch.LongTensor(item_inputs))
        element_embeds = torch.mul(user_embeds,item_embeds)
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

