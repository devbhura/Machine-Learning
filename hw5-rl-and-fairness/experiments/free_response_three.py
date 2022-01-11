import gym
from src import MultiArmedBandit, QLearning
import matplotlib.pyplot as plt
import numpy as np

# FR3a

env = gym.make('FrozenLake-v1')

rewards_array = []
for i in range(10):
    agent = QLearning(epsilon=0.01)
    action_values, rewards = agent.fit(env,steps=100000)
    rewards_array.append(rewards)


avg_10_trials = np.zeros(100)

for i in range(10):
    avg_10_trials += rewards_array[i]

avg_rewards_low_eps = avg_10_trials/10

plt.figure()
plt.plot(np.linspace(0,99,100),avg_rewards_low_eps,label='eps = 0.01')

# FR3b
rewards_array = []
for i in range(10):
    agent = QLearning(epsilon=0.5)
    action_values, rewards = agent.fit(env,steps=100000)
    rewards_array.append(rewards)

avg_10_trials = np.zeros(100)

for i in range(10):
    avg_10_trials += rewards_array[i]

avg_rewards_high_eps = avg_10_trials/10
plt.plot(np.linspace(0,99,100),avg_rewards_high_eps,label='eps=0.5')

plt.legend()
plt.title('QLearning with different epsilon - Exploration vs Exploitation')
plt.xlabel('rewards step')
plt.ylabel('rewards')
plt.savefig('images/FR3.png')

