import gym
import numpy as np


def test_bandit_slots():
    """
    Tests that the MultiArmedBandit implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from src import MultiArmedBandit

    np.random.seed(0)

    env = gym.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    means = np.array([m.mean for m in env.machines])

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=10000)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)
    

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(states) == 1
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(rewards) == 1


def test_bandit_frozen_lake():
    """
    Tests the MultiArmedBandit implementation on the FrozenLake-v0 environment.
    """
    from src import MultiArmedBandit

    np.random.seed(0)

    env = gym.make('FrozenLake-v1')
    env.seed(0)

    agent = MultiArmedBandit(epsilon=0.2)
    state_action_values, rewards = agent.fit(env, steps=10000)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100
