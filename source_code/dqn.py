import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Lambda, Add
from keras.models import Model
from keras import backend as K
from enviroment import Environment
from parameters import *
from util.csv_util import *

class DQN:
  def __init__(self, dueling=False, power=0):
    self.env = Environment()
    self.dueling = dueling

    self.action_history = []
    self.state_history = []
    self.reward_history = []
    self.next_state_history = []

    self.model = self.create_model(dueling)
    self.target_model = self.create_model(dueling)

    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.9999
    self.loss_function = tf.keras.losses.Huber()
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_deepQ)

    self.rewards = []

    # Custom params
    self.power = power
    self.file_name = 'ddqn_' + str(self.power) + 'W.csv' if dueling else 'dqn_' + str(self.power) + 'W.csv'


  def create_model(self, dueling):
    input_shape = (num_features,)
    X_input = Input(input_shape) # Input_layer
    X = X_input

    # First hidden layer
    X = Dense(512, input_shape=input_shape, activation="tanh")(X)
    # Second hidden layer
    X = Dense(256, activation="tanh")(X)
    # Third hidden layer
    X = Dense(64, activation="tanh")(X)
    
    if dueling:
        # Dueling DQN architecture
        # Value Stream
        V = Dense(16, activation="tanh")(X)
        V = Dense(1, activation="linear")(V)

        # Advantage Stream
        A = Dense(16, activation="tanh")(X)
        A = Dense(num_actions, activation="linear")(A)

        # Combine Value and Advantage streams
        Q_values = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)))([V, A])
    else:
        # Standard DQN architecture
        Q_values = Dense(num_actions, activation="linear")(X)

    model = Model(inputs = X_input, outputs = Q_values)
    return model


  def remember(self, state, action, reward, next_state):
    self.state_history.append(state)
    self.action_history.append(action)
    self.reward_history.append(reward)
    self.next_state_history.append(next_state)

    if len(self.reward_history) > memory_size:
      del self.state_history[:1]
      del self.action_history[:1]
      del self.reward_history[:1]
      del self.next_state_history[:1]


  def replay(self):
    indices = np.random.choice(range(len(self.reward_history)), size=batch_size)
    state_sample = np.array([self.state_history[i] for i in indices]).reshape((batch_size, num_features))
    action_sample = [self.action_history[i] for i in indices]
    reward_sample = [self.reward_history[i] for i in indices]
    next_state_sample = np.array([self.next_state_history[i] for i in indices]).reshape((batch_size, num_features))
    future_rewards = self.target_model.predict(next_state_sample, verbose=0)
    updated_q_values = reward_sample + gamma_deepQ * tf.reduce_max(future_rewards, axis=1)
    masks = tf.one_hot(action_sample, num_actions)

    with tf.GradientTape() as tape:
      q_values = self.model(state_sample, training=False)
      q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
      loss = self.loss_function(updated_q_values, q_action)

    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


  def target_update(self):
    self.target_model.set_weights(self.model.get_weights())


  def get_action(self, state):
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon_min, self.epsilon)
    if np.random.random() < self.epsilon:
      return np.random.choice(self.env.get_possible_action())
    else:
      list_value = self.model.predict(state, verbose=0)[0]
      list_actions = self.env.get_possible_action()
      max_q = -float("inf")
      action = 0
      for action_t in list_actions:
        if list_value[action_t] >= max_q:
          max_q = list_value[action_t]
          action = action_t
    return action


  def learning(self):
    create_csv(self.file_name, 'Iteration', 'Avg Throughput')

    total_reward = 0
    for i in range(T):
      current_state = self.env.get_state_deep()
      current_state = np.reshape(current_state, (1, num_features))
      action = self.get_action(current_state)

      reward, next_state = self.env.perform_action_deep(action)
      next_state = np.reshape(next_state, (1, num_features))
      total_reward += reward
      self.remember(current_state, action, reward, next_state)

      self.replay()

      # append rewards for plot graph
      self.rewards.append(total_reward / (i + 1))

      if (i + 1) % update_target_network == 0:
        self.target_update()
      if (i + 1) % step == 0:
        print("Iteration " + str(i + 1) + " reward: " + str(total_reward / (i + 1)))
        insert_row(self.file_name, i + 1, total_reward / (i + 1))


    # save model
    # if self.dueling:
    #   self.model.save('model/ddqn.keras')
    #   np.save('model/ddqn.npy', self.rewards)
    # else:
    #   self.model.save('model/dqn.keras')
    #   np.save('model/dqn.npy', self.rewards)

  def save_model(self):
    # save model
    if self.dueling:
      name = 'model/ddqn_' + str(self.power) + 'W.keras'
      self.model.save(name)
      # np.save('model/ddqn.npy', self.rewards)
    else:
      name = 'model/dqn_' + str(self.power) + 'W.keras'
      self.model.save(name)
      # np.save('model/dqn.npy', self.rewards)
    