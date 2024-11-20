import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment import Environment
from parameters import *
import random
import numpy as np
from scipy.stats import poisson
from enviroment import JammerState
from util.csv_util import *

class GreedyStrategy:
    def __init__(self, d_t=d_t):
        self.env = Environment()
        self.success_package_num = 0
        self.package_lost = 0
        self.total_packages_sent = 0

        # Custom params
        self.d_t = d_t
        self.nu = nu
        self.nu_p = nu_p
        self.harvest_frequency = 5

    def set_jammer_power(self, nu, nu_p):
        self.nu = nu
        self.nu_p = nu_p
        # self.env.set_jammer_power(nu=nu, nu_p=nu_p)

    def can_transmit(self):
        return self.env.energy_state >= e_t
    
    def harvest_energy(self):
        harvest_energy = random.choices(e_hj_arr, weights=self.nu_p, k=1)[0]
        self.env.energy_state += harvest_energy

        if self.env.energy_state > e_queue_size:
            self.env.energy_state = e_queue_size

    def active_transmit(self):
        package_transmit_success = self.env.active_transmit(self.d_t)
        self.env.data_state -= package_transmit_success
        self.env.energy_state -= package_transmit_success * e_t
        self.success_package_num += package_transmit_success

    def backscatter(self):
        d_bj = random.choices(d_bj_arr, weights=self.nu_p, k=1)[0]
        
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
        T = 5000

        for i in range(T):
            if self.env.jammer_state == JammerState.IDLE.value:
                self.env.active_transmit(self.d_t)
            else:
                if (i + 1) % self.harvest_frequency == 0:
                    # harvest energy
                    self.harvest_energy()
                else:
                    # backscatter
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
                if np.random.random() <= 1 - self.nu:
                    self.env.jammer_state = JammerState.ATTACK.value
            else:
                if np.random.random() <= self.nu:
                    self.env.jammer_state = JammerState.IDLE.value

        # Print result
        print('---------------------------------------------------')
        print('Result after running simulation in ' + str(T) + ' time units')
        # print('Total rewards = ' + str(total_reward))
        print('Number packages sent successfully = ' + str(self.success_package_num))
        print('Avg throughput (packages/time unit) = ' + str(self.success_package_num / T))
        print('Avg loss (packages/time unit) = ' + str(self.package_lost / T))
        print('PDR = ' + str(self.success_package_num / self.total_packages_sent * 100) + '%') 
        # print('---------------------------------------------------')
        # print('Total packages send = ' + str(self.total_packages_sent))
        # print('Loss packages = ' + str(self.package_lost))
        # print('Success packages = ' + str(self.success_package_num))
        # print('Package still in queue = ' + str(self.env.data_state))

        # Print result to csv
        # self.print_result_power(self.success_package_num / T, self.package_lost / T, self.success_package_num / self.total_packages_sent * 100)
        return self.success_package_num / T, self.package_lost / T, self.success_package_num / self.total_packages_sent * 100

    def print_result(self, through_put, package_loss, pdr):
        create_csv('greedy_throughput.csv', 'd_t', 'throughput')
        create_csv('greedy_loss.csv', 'd_t', 'package_loss')
        create_csv('greedy_pdr.csv', 'd_t', 'pdr')
        insert_row('greedy_throughput.csv', self.d_t, through_put)
        insert_row('greedy_loss.csv', self.d_t, package_loss)
        insert_row('greedy_pdr.csv', self.d_t, pdr)

    def print_result_power(self, through_put, package_loss, pdr):
        create_csv('greedy_power_throughput.csv', 'P_avg', 'throughput')
        create_csv('greedy_power_loss.csv', 'P_avg', 'package_loss')
        create_csv('greedy_power_pdr.csv', 'P_avg', 'pdr')
        insert_row('greedy_power_throughput.csv', self.power, through_put)
        insert_row('greedy_power_loss.csv', self.power, package_loss)
        insert_row('greedy_power_pdr.csv', self.power, pdr)