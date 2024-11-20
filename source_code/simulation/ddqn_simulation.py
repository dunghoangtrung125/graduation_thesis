import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment import Environment
import numpy as np
import tensorflow as tf
from parameters import *
from keras import backend as K
from util.csv_util import *

class DDQNSimulate:
    def __init__(self, model='model/ddqn.keras', time_units=5000):
        self.env = Environment()
        self.time_units = time_units
        self.model = tf.keras.models.load_model(model, custom_objects={'K': K})

    def set_jammer_power(self, nu=nu, nu_p=nu_p):
        self.env.set_jammer_power(nu=nu, nu_p=nu_p)
    
    def set_active_transmit_packages(self, d_t=d_t):
        self.env.set_active_transmission_package_num(d_t)

    def get_action(self):
        return np.random.choice(self.env.get_possible_action())
    
    def run(self):
        total_reward = 0
        for i in range(self.time_units):
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


        print('---------------------------------------------------')
        print('Result after running simulation in ' + str(self.time_units) + ' time units')
        print('Total rewards = ' + str(total_reward))
        print('Number packages sent successfully = ' + str(self.env.total_packages_arrival - self.env.loss_packages))
        print('Avg throughput (packages/time unit) = ' + str(total_reward / self.time_units))
        print('Avg loss (packages/time unit) = ' + str(self.env.loss_packages / self.time_units))
        print('PDR = ' + str((self.env.total_packages_arrival - self.env.loss_packages) / self.env.total_packages_arrival * 100) + '%') 
        print('---------------------------------------------------')
        # print('---------------------------------------------------')
        # print('Package still in queue = ' + str(self.env.data_state))
        # print('Success packages = ' + str(total_reward))
        # print('Loss packages = ' + str(self.env.loss_packages))
        # print('Total packages arrival = ' + str(self.env.total_packages_arrival))
        # self.print_result(total_reward / T, self.env.loss_packages / T, (self.env.total_packages_arrival - self.env.loss_packages) / self.env.total_packages_arrival * 100)
        return total_reward / self.time_units, self.env.loss_packages / self.time_units, (self.env.total_packages_arrival - self.env.loss_packages) / self.env.total_packages_arrival * 100

