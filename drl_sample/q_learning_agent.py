import numpy as np
from enviroment import Environment
from parameters import *

class q_learning_agent:
  def __init__(self):
    self.env = Environment()
    self.q_matrix = np.zeros((num_states, num_actions))
    self.rewards = []

  def learning(self):
    epsilon = 1
    decay = 0.9999
    min_epsilon = 0.01
    action = 0
    total_reward = 0
    for i in range(T):
      current_state = self.env.get_state()
      list_possible_action = self.env.get_possible_action()
      max_q = -float("inf")
      if np.random.random() <= epsilon:
        action = np.random.choice(list_possible_action)
      else:
        for action_t in list_possible_action:
          if self.q_matrix[current_state][action_t] > max_q:
            max_q = self.q_matrix[current_state][action_t]
            action = action_t

      reward, next_state = self.env.perform_action(action)
      total_reward += reward
      self.rewards.append(total_reward / (i + 1))
      list_possible_next_state = self.env.get_possible_action()
      max_q_next = -float("inf")
      for action_n in list_possible_next_state:
        if np.all(self.q_matrix[next_state][action_n] >= max_q):
          max_q = self.q_matrix[next_state][action_n]

      data = (1 - learning_rate_Q) * self.q_matrix[current_state][action] + learning_rate_Q * (reward + gamma_Q * max_q)
      self.q_matrix[current_state][action] = data
      temp = epsilon * decay
      epsilon = max(min_epsilon, temp)
      if (i + 1) % step == 0:
        print("Iteration " + str(i + 1) + " reward: " + str(total_reward / (i + 1)))