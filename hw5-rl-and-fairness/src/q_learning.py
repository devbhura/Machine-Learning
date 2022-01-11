import numpy as np


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      alpha - (float) The weighting to give current rewards in estimating Q. This 
        should range [0,1], where 0 means "don't change the Q estimate based on 
        current reward" 
      gamma - (float) This is the weight given to expected future rewards when 
        estimating Q(s,a). It should be in the range [0,1]. Here 0 means "
        don't incorporate estimates of future rewards into the reestimate of
        Q(s,a)"
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
        
      See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode and for definitions of alpha, gamma, epsilon 
        (http://incompleteideas.net/book/RLbook2020.pdf).  
    """

    def __init__(self, epsilon=0.2, alpha=.5, gamma=.5, adaptive=False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. Note that this is a different formula
        for the step size than was used in MultiArmedBandits. Use an
        epsilon-greedy policy for action selection. Note that unlike the
        pseudocode, we are looping over a total number of steps, and not a
        total number of episodes. This allows us to ensure that all of our
        trials have the same number of steps--and thus roughly the same amount
        of computation time.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - Use the provided self._get_epsilon function whenever you need to
            obtain the current value of epsilon.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        s = env.observation_space.n
        a = env.action_space.n
        state_action_values = np.random.rand(s,a)
        # state_action_values = 0.05*np.ones((s,a))
        n_a = np.zeros(a)
        rewards = np.zeros(100)
        actions = np.arange(a)
        state = env.reset()

        reward_s = np.floor(steps/100)
        reward_storing = []

        for step in range(steps):
          env.render()

          if self.adaptive:
            self.epsilon = self._get_epsilon(step/steps)
          

          if np.random.random() >= self.epsilon:
            action = np.argmax(actions)
          else:
            action = np.random.choice(len(actions))

          n_a[action] += 1
          
          next_state, reward, done, info = env.step(action)

          if state!=next_state and done:
            state_action_values[next_state,:] = 0
            # print(f"""
            # state = {state}
            # next_state = {next_state}
            # reward = {reward}
            # done = {done}
            # action = {action}
            # state_action_val = {state_action_values}
            # """)

          state_action_values[state][action] += self.alpha*(reward +self.gamma*(np.max(state_action_values[next_state,:])) - state_action_values[state][action])
          
          # print(f"""
          # state = {state}
          # next_state = {next_state}
          # reward = {reward}
          # done = {done}
          # action = {action}
          # state_action_val = {state_action_values[state][action]}
          # """)

          # if step == 2:
          #   raise ValueError
          reward_storing.append(reward)
          
          if step % reward_s == 0:
            rewards[int(step/reward_s)] = np.mean(np.array(reward_storing))
            reward_storing = []



          if done:
            
            state = env.reset()
            
          else:
            state = next_state

        
        env.close()
        return state_action_values, rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state
          - You should use a loop to predict over each step in an episode until
            it terminates; see /src/slot_machines.py for an example of how an
            environment signals the end of an episode using the step() method

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        states = []
        actions = []
        rewards = []

        
        done = False 
        state = env.reset()
        # state_action_values, re = self.fit(env)
        while not done:
          
          action = np.argmax(state_action_values[state,:])
          
          state, reward, done, info = env.step(action)
          # print(f"""
          # state = {state}
          # reward = {reward}
          # done = {done}
          # """)

          states.append(state)
          actions.append(action)
          rewards.append(reward)



        return np.array(states), np.array(actions), np.array(rewards)

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        See free response question 3 for instructions on how to implement this
        function.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return (1-progress)*self.epsilon
