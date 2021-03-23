
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, n_feature, n_action, n_hidden=20):
        super(Actor, self).__init__()

        self.input_dim = n_feature
        self.output_dim = n_action

        self.l1 = nn.Linear(self.input_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden, self.output_dim)


    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = torch.softmax(x, dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, n_feature, value_num=1, n_hidden=20):
        super(Critic, self).__init__()

        self.input_dim = n_feature
        self.output_dim = value_num

        self.l1 = nn.Linear(self.input_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden, self.output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

