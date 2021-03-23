
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from net import policy_net

np.random.seed(1)
torch.manual_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr = self.lr)



    def _build_net(self):
        self.model = policy_net(
            self.n_features, self.n_actions
        )

    def choose_action(self, observation):
        prob_weights = self.model(torch.Tensor(observation[np.newaxis, :]))
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.clone().detach().numpy().ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # train on episode
        action_prob = self.model(torch.Tensor(np.vstack(self.ep_obs)))
        action_onehot = F.one_hot(torch.LongTensor(self.ep_as), num_classes=self.n_actions)
        reward_discounted = discounted_ep_rs_norm

        self.optimizer.zero_grad()
        self.loss = torch.mean(
            torch.sum((-1 * torch.log(action_prob) * action_onehot), dim=1) *reward_discounted)
        self.loss.backward()
        print(self.loss.item())
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount eposide rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return torch.Tensor(discounted_ep_rs)