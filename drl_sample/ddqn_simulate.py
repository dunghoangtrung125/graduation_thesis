from enviroment import Environment
import numpy as np
import tensorflow as tf
from parameters import *

class DDQNSimulate:
    def __init__(self):
        self.env = Environment()
        self.model = tf.keras.models.load_model('model/ddqn.keras')

    def get_action(self):
        return np.random.choice(self.env.get_possible_action())
    
    def run(self):
        T = 1_000_000
        total_reward = 0
        for i in range(T):
            state = self.env.get_state_deep()
            state = np.reshape(state, (1, num_features))
            list_value = self.model.predict(state, verbose=0)[0]
            list_actions = self.env.get_possible_action()

            max_q = -float("inf")
            action = 0
            for action_t in list_actions:
                if list_value[action_t] >= max_q:
                    max_q = list_value[action_t]
                    action = action_t

            reward, state = self.env.perform_action_deep(action)
            total_reward += reward
            if (i + 1) % step == 0:
                print('Reward at ' + str(i + 1) + ' is ' + str(total_reward / (i+1)))

