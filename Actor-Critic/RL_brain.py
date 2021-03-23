
from net import Actor, Critic
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

class Actor_Critic:
    def __init__(self, n_feature, n_action, lr_A=0.001, lr_C=0.01, GAMMA=0.1):
        self.n_feature = n_feature
        self.n_action = n_action
        self.GAMMA = GAMMA

        self.actor = Actor(self.n_feature, self.n_action)
        self.critic = Critic(self.n_feature)

        self.optimizer_actor = optim.Adam(params=self.actor.parameters(),
                                          lr=lr_A)
        self.optimizer_critic = optim.Adam(params=self.critic.parameters(),
                                          lr=lr_C)

        self.cost_his = []
        self.value_his = []


    def actor_learn(self, s, a, td_error):
        s = torch.Tensor(s[np.newaxis, :])  ##batch=1

        action_prob = self.actor(s) ##[batch x self.n_action]-->[1, n_action]
        log_prob = torch.log(action_prob[0, a]) ## a in action index

        self.exp_v = torch.mean(-1*log_prob*td_error)

        self.optimizer_actor.zero_grad()
        self.exp_v.backward()
        self.optimizer_actor.step()

        self.value_his.append(self.exp_v.item())
        return self.exp_v

    def choose_action(self, s):
        s = torch.Tensor(s[np.newaxis,:])
        probs = self.actor(s)
        return np.random.choice(range(probs.shape[1]), p=probs.clone().detach().numpy().ravel())


    def critic_learn(self, s, r, s_):
        s, s_ = torch.Tensor(s[np.newaxis,:]), torch.Tensor(s_[np.newaxis,:])

        v_ = self.critic(s_) ##part of Q target
        v = self.critic(s)
        td_error = F.mse_loss(v, r + self.GAMMA * v_)

        self.cost_his.append(td_error.item())

        self.optimizer_critic.zero_grad()
        td_error.backward()
        self.optimizer_critic.step()

        return td_error.item()

    def plot_cost(self):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2,1)

        ax1.plot(np.arange(len(self.cost_his)), self.cost_his)
        ax1.set_ylabel('Critic TD error')

        ax2.plot(np.arange(len(self.value_his)), self.value_his)
        ax2.set_ylabel('Actor value')
        ax2.set_xlabel('training steps')

        plt.show()

        pass

