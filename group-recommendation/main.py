'''
Created on Nov 10, 2017
Main function

@author: Lianhai Miao
'''
from model.agree import AGREE
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset


# train the model
def training(model, train_loader, epoch_id, config, type_m):
    lr = config.lr
    optimizer = optim.Adam(model.parameters(), lr)

    losses = 0
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward      모델을 2개돌리고 합쳐서 loss로 한다
        if type_m == 'user':
            pos_pred = model(None, user_input, pos_item_input)
            neg_pred = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_pred = model(user_input, None, pos_item_input)
            neg_pred = model(user_input, None, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_pred - neg_pred -1) **2)
        # record loss history
        losses += float(loss)  
        # Backward
        loss.backward()
        optimizer.step()

    losses /= batch_id
    print('Iteration %d, loss is [%.4f ]' % (epoch_id, losses))

if __name__ == '__main__':
    config = Config()
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

    # get group number
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items

    # build AGREE model
    agree = AGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, config.drop_ratio)

    # config information
    print("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" %(config.embedding_size, config.epoch, config.topK))
    # train the model
    for epoch in range(config.epoch):
        agree.train()
        # training
        t1 = time()
        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')
        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("user and group training time is: [%.1f s]" % (time()-t1))
    print("Done!")
