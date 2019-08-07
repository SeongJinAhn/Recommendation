import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio,gpu_available):
        super(AGREE, self).__init__()
        self.userembeds = nn.Embedding(num_users, embedding_dim)
        self.itemembeds = nn.Embedding(num_items, embedding_dim)
        self.groupembeds = nn.Embedding(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)  #member, item 64
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio) #member, item. group 96
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        self.gpu_available = gpu_available

        # initial model         배워갑니다
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
        _Tensor = torch.LongTensor
        if self.gpu_available == True:
            device = torch.device('cuda')
            _Tensor = torch.cuda.LongTensor
            group_inputs = group_inputs.to(device)
            item_inputs = item_inputs.to(device)

        group_embeds = torch.Tensor()
        item_embeds = self.itemembeds(_Tensor(item_inputs))

        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[int(i)]
            members_embeds = self.userembeds(_Tensor(members))
            each_item_embeds = self.itemembeds(_Tensor([j for _ in members]))

            group_item_embeds = torch.cat((members_embeds, each_item_embeds), dim=1)         #group을 개인의 영향력들의 linear sum으로 embeds
            at_wt = self.attention(group_item_embeds)                                   # input dim : [2, 64]
            g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            group_embeds = torch.cat((group_embeds, g_embeds_with_attention))

        element_embeds = torch.mul(group_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, group_embeds, item_embeds), dim=1)
        y = F.sigmoid(self.predictlayer(new_embeds))
        return y

    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        if self.gpu_available == True:
            user_inputs = user_inputs.to(torch.device('cuda'))
            item_inputs = item_inputs.to(torch.device('cuda'))

        user_embeds = self.userembeds(user_inputs)
        item_embeds = self.itemembeds(item_inputs)
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = F.sigmoid(self.predictlayer(new_embeds))
        return y


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
