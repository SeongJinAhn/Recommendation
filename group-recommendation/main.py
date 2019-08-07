import torch
import torch.nn as nn
import torch.optim as optim
from model.agree import AGREE
import numpy as np
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
        if config.GPU_available == True:
            u = u.to(device)
            pi_ni = pi_ni.to(device)

        user_input = u
        pos_item, neg_item = pi_ni[:,0], pi_ni[:,1]
        # Forward      모델을 2개돌리고 합쳐서 loss로 한다
        if type_m == 'user':
            pos_pred = model(None, user_input, pos_item)
            neg_pred = model(None, user_input, neg_item)
        elif type_m == 'group':
            pos_pred = model(user_input, None, pos_item)
            neg_pred = model(user_input, None, neg_item)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_pred - neg_pred -1) **2)
        losses += float(loss)  
        # Backward
        loss.backward()
        optimizer.step()

    losses /= batch_id
    print('Iteration %d, loss is [%.4f ]' % (epoch_id, losses))

def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg

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
    agree = AGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, config.drop_ratio, config.GPU_available)

    # config information
    print("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" %(config.embedding_size, config.epoch, config.topK))
    # train the model
    if config.GPU_available:
        device = torch.device('cuda')
        agree = agree.to(device)

    for epoch in range(config.epoch):
        agree.train()
        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')
        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')

        u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user')
        hr, ndcg = evaluation(agree, helper, dataset.group_testRatings, dataset.group_tyestNegatives, config.topK, 'group')
    print("Done!")
