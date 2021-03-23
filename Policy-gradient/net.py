
import torch
import torch.nn as nn
import torch.nn.functional as F

class policy_net(nn.Module):
    def __init__(self, n_features, n_action, n_hidden=10):

        super(policy_net, self).__init__()
        self.input_dim = n_features
        self.output_dim = n_action
        self.hidden_dim = n_hidden

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):

        x = self.l1(x)
        x = torch.tanh(x)
        x = self.l2(x)
        x = F.softmax(x, dim=1)
        return x



def test():
    input = torch.rand(6, 4)
    model = policy_net(4, 7, 10)
    output = model(input)
    print(torch.sum(output[0,:]))

if __name__=="__main__":
    test()