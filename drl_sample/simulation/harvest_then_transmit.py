import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment import Environment
from parameters import *
import random
import numpy as np
from scipy.stats import poisson
from enviroment import JammerState

class HarvestThenTransmitt:

    def __init__(self):
        self.env = Environment()
        self.success_package_num = 0
        self.package_lost = 0
        self.total_packages_sent = 0

    def can_transmit(self):
        return self.env.energy_state >= e_t
    
    def harvest_energy(self):
        harvest_energy = random.choices(e_hj_arr, weights=nu_p, k=1)[0]
        self.env.energy_state += harvest_energy

        if self.env.energy_state > e_queue_size:
            self.env.energy_state = e_queue_size

    def active_transmit(self):
        package_transmit_success = self.env.active_transmit(d_t)
        self.env.data_state -= package_transmit_success
        self.env.energy_state -= package_transmit_success * e_t
        self.success_package_num += package_transmit_success

    def run(self):
        T = 40_000
        for i in range(T):
            if self.env.jammer_state == JammerState.IDLE.value:
                # Active transmit
                self.active_transmit()
            else:
                # Jammer is attacking, HTT
                self.harvest_energy()

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
        print('---------------------------------------------------')
        print('Total packages send = ' + str(self.total_packages_sent))
        print('Loss packages = ' + str(self.package_lost))
        print('Success packages = ' + str(self.success_package_num))
        print('Package still in queue = ' + str(self.env.data_state))
