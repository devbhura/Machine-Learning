import gym
from src import MultiArmedBandit,QLearning
import matplotlib.pyplot as plt
import numpy as np

# print('Starting example experiment')

# env = gym.make('FrozenLake-v1')
# agent = MultiArmedBandit()
# action_values, rewards = agent.fit(env)

# print('Finished example experiment')


# FR 2a 
env = gym.make('SlotMachines-v0')

rewards_array = []
for i in range(10):
    agent = MultiArmedBandit()
    action_values, rewards = agent.fit(env,steps=100000)
    rewards_array.append(rewards)

plt.figure()
plt.plot(np.linspace(0,99,100),rewards_array[0],label='First trial')

avg_5_trials = np.zeros(100)

for i in range(5):
    avg_5_trials += rewards_array[i]

avg_5_trials = avg_5_trials/5

plt.plot(np.linspace(0,99,100),avg_5_trials,label='First 5 trials')


avg_10_trials = np.zeros(100)

for i in range(10):
    avg_10_trials += rewards_array[i]

avg_10_trials = avg_10_trials/10

plt.plot(np.linspace(0,99,100),avg_10_trials,label='First 10 trials')

plt.legend()
plt.title('Multi-Armed Bandit on Slot Machines')
# plt.show()
plt.xlabel('Rewards Step')
plt.ylabel('Rewards')
plt.savefig('images/MAB_SM.png')

# FR2b
env = gym.make('SlotMachines-v0')

rewards_array = []
for i in range(10):
    agent = QLearning()
    action_values, rewards = agent.fit(env,steps=100000)
    rewards_array.append(rewards)

plt.figure()
plt.plot(np.linspace(0,99,100),rewards_array[0],label='First trial')

avg_5_trials = np.zeros(100)

for i in range(5):
    avg_5_trials += rewards_array[i]

avg_5_trials = avg_5_trials/5

plt.plot(np.linspace(0,99,100),avg_5_trials,label='First 5 trials')


avg_10_trials = np.zeros(100)

for i in range(10):
    avg_10_trials += rewards_array[i]

avg_10_trials = avg_10_trials/10

plt.plot(np.linspace(0,99,100),avg_10_trials,label='First 10 trials')

plt.legend()
plt.title('Q-learning on Slot Machines')
# plt.show()
plt.xlabel('Rewards Step')
plt.ylabel('Rewards')
plt.savefig('images/QL_SM.png')