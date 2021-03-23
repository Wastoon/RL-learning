
import gym
from RL_brain import Actor_Critic

import matplotlib.pyplot as plt

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

actor_critic = Actor_Critic(N_F, N_A, lr_A=LR_A, lr_C=LR_C, GAMMA=GAMMA)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []

    while True:
        if RENDER: env.render()

        a = actor_critic.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = actor_critic.critic_learn(s, r, s_)
        actor_critic.actor_learn(s, a, td_error)

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = False  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

actor_critic.plot_cost()

