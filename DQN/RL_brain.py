
import numpy as np
from net import EvaluateNet, TargetNet, freeze_parameter
import torch.optim as optim
import torch

class DeepQNetwork:

    def __init__(self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.001,
            output_graph=False,):


        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max


        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.cost_his = []

        self.optimizer = optim.RMSprop(params=self.evaluate_net.parameters(),
                                       lr=self.lr)

        self.loss = torch.nn.MSELoss()


    def _build_net(self):
        self.evaluate_net = EvaluateNet(self.n_features, self.n_actions)
        self.targetnet    = TargetNet(self.n_features, self.n_actions)
        self.targetnet = freeze_parameter(self.targetnet)


    def store_transtion(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s,[a,r],s_))

        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
        pass

    def choose_action(self, observation):
        # to have batch dimension when feed into model
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            action_value = self.evaluate_net(torch.Tensor(observation))
            action = torch.argmax(action_value)
        else:
            action = torch.tensor(np.random.randint(0, self.n_actions), dtype=torch.int32)
        return action

        pass

    def _replace_target_params(self):
        print('\ntarget_params_replaced\n')
        for parameter_new, parameter_old in zip(self.evaluate_net.parameters(), self.targetnet.parameters()):
            parameter_old.data = parameter_new.data
        pass

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = torch.Tensor(self.memory[sample_index, :])

        q_next = self.targetnet(batch_memory[:, -self.n_features:])
        q_eval = self.evaluate_net(batch_memory[:, :self.n_features])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        batch_index = torch.arange(self.batch_size, dtype=torch.int32).type(torch.LongTensor)
        eval_act_index = batch_memory[:, self.n_features].type(torch.LongTensor)
        reward = batch_memory[:, self.n_features+1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        #train eval network
        self.loss_ = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        self.loss_.backward()
        print(self.loss_.item())
        self.optimizer.step()

        self.cost_his.append(self.loss_.item())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        pass

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

        pass
