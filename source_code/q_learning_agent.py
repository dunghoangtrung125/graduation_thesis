from enviroment import Environment
import numpy as np
from parameters import *

class QLearningAgent:
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

    for i in range(1_000_000):
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
      list_possible_next_action = self.env.get_possible_action()
      max_q = -float("inf")
      for action_n in list_possible_next_action:
        if self.q_matrix[next_state][action_n] >= max_q:
          max_q = self.q_matrix[next_state][action_n]

      data = (1 - learning_rate_Q) * self.q_matrix[current_state][action] + learning_rate_Q * (reward + gamma_Q * max_q)
      self.q_matrix[current_state][action] = data
      temp = epsilon * decay
      epsilon = max(min_epsilon, temp)
      self.rewards.append(total_reward / (i + 1))
      if (i + 1) % step == 0:
        print("Iteration " + str(i + 1) + " reward: " + str(total_reward / (i + 1)))
    
    self.save_model('model/q_matrix.npy')


  def save_model(self, file_path):
    np.save(file_path, self.q_matrix)
    print(f"Model saved to {file_path}")

  def load_model(self, file_path):
    self.q_matrix = np.load(file_path)
    print(f"Model loaded from {file_path}")

  def predict(self, state):
        # Get possible actions for the given state
        possible_actions = self.env.get_possible_action()
        
        # Select the action with the highest Q-value for this state
        best_action = max(
            possible_actions, 
            key=lambda action: self.q_matrix[state][action]
        )
        
        return best_action

