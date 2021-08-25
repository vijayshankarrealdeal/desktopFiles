import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

n_games = 1000
win_per = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = env.action_space.sample()
        obs,reward,done,info = env.step(action)
        score += reward
    scores.append(score)
    if i %10 == 0:
        win_per.append(np.mean(scores[-10:]))
        
plt.plot(win_per)
plt.show()
    