import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment import Environment
from parameters import *
import random
import numpy as np
from scipy.stats import poisson
from enviroment import JammerState

class BackScatterStrategy:

    def __init__(self):
        self.env = Environment()
        self.success_package_num = 0
        self.package_lost = 0
        self.total_packages_sent = 0
    
    def backscatter(self):
        d_bj = random.choices(d_bj_arr, weights=nu_p, k=1)[0]
        
        max_rate = min(self.env.data_state, b_dagger)
        reward = min(d_bj, self.env.data_state)

        if max_rate > reward:
            # jammer back scatter ability smaller than b_dagger
            self.package_lost += max_rate - reward
            self.success_package_num += reward
        else:
            self.success_package_num += max_rate

        self.env.data_state -= max_rate

    def run(self):
        T = 40_000
        for i in range(T):
            if self.env.jammer_state == JammerState.IDLE.value:
                # Do nothing
                pass
            else:
                # Jammer is attacking, use backscatter technique
                self.backscatter()

            # data arrival
            data_arrive_l = poisson.rvs(mu=arrival_rate, size=1)
            data_arrive = data_arrive_l[0]
            self.env.data_state += data_arrive
            self.total_packages_sent += data_arrive
            if self.env.data_state > d_queue_size:
                self.package_lost += self.env.data_state - d_queue_size
                self.env.data_state = d_queue_size

            # jammer state
            if self.env.jammer_state == JammerState.IDLE.value:
                if np.random.random() <= 1 - nu:
                    self.env.jammer_state = JammerState.ATTACK.value
            else:
                if np.random.random() <= nu:
                    self.env.jammer_state = JammerState.IDLE.value
        
        # Print result
        print('---------------------------------------------------')
        print('Result after running simulation in ' + str(T) + ' time units')
        # print('Total rewards = ' + str(total_reward))
        print('Number packages sent successfully = ' + str(self.success_package_num))
        print('Avg throughput (packages/time unit) = ' + str(self.success_package_num / T))
        print('Avg loss (packages/time unit) = ' + str(self.package_lost / T))
        print('PDR = ' + str(self.success_package_num / self.total_packages_sent * 100) + '%') 
        print('---------------------------------------------------')
        print('Total packages send = ' + str(self.total_packages_sent))
        print('Loss packages = ' + str(self.package_lost))
        print('Success packages = ' + str(self.success_package_num))
        print('Package still in queue = ' + str(self.env.data_state))
        print(d_t)