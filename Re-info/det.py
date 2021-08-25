import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')

n_games = 1000
win_pre = []
scores = []

left = 0
down = 1
right = 2
up = 3

'''
SFFF
FHFH
FFFH
HFFG
'''

policy = {0:1,1:2,2:1,3:0,4:1,6:1,8:2,9:1,10:1,13:2,14:2}

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = policy[obs]
        obs,reward,done,info = env.step(action)
        score += reward
    scores.append(score)
    if i %10 == 0:
        win_pre.append(np.mean(scores[-10:]))
        
        
        
plt.plot(win_pre)
plt.show()
        
        
        
        
        
        
        
    