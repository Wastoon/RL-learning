import torch
import torch.nn as nn
import torch.nn.functional as F


##evaluate network
class EvaluateNet(nn.Module):
    def __init__(self, n_feature, n_action, n_hidden=10):
        super(EvaluateNet, self).__init__()
        self.input_dim = n_feature
        self.output_dim = n_action
        self.hidden_dim = n_hidden

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.l2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def forward(self, x):
        ## x: [batch_size x n_feature]
        x = self.l1(x)
        x = self.l2(x)
        x = F.relu(x)
        return x


##target network
class TargetNet(nn.Module):
    def __init__(self, n_feature, n_action, n_hidden=10):
        super(TargetNet, self).__init__()
        self.input_dim = n_feature
        self.output_dim = n_action
        self.hidden_dim = n_hidden

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.l2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)


    def forward(self, x):
        ## x: [batch_size x n_feature]
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x


def freeze_parameter(model):
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model

def test():
    input = torch.rand(6, 4)
    model = EvaluateNet(4, 7, 10)
    model1 = TargetNet(4, 7, 10)
    output = model(input)
    #print(output.shape)
    import pdb
    pdb.set_trace()
    for parameter_name, parameter in model.named_parameters():
        print(parameter)

    freeze_parameter(model1)
    for parameter_name, parameter in model1.named_parameters():
        print(parameter)

    for parameter_new, parameter_old in zip(model.parameters(), model1.parameters()):
        parameter_old.data = parameter_new.data

    for parameter_name, parameter in model1.named_parameters():
        print(parameter)
if __name__=="__main__":
    test()


