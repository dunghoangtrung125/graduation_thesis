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
        self.time = 1_000_000
        self.package_lost = 0

    def can_transmit(self):
        return self.env.energy_state >= e_t
    
    def harvest_energy(self):
        harvest_energy = random.choices(e_hj_arr, weights=nu_p, k=1)[0]
        self.env.energy_state += harvest_energy

        if self.env.energy_state > e_queue_size:
            self.env.energy_state = e_queue_size

    def transmit_data(self):
        max_ra = d_t # when jammer idle
        if self.env.jammer_state == JammerState.ATTACK:
            max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
        
        package_transmit_success = self.env.active_transmit(max_ra)
        self.env.data_state -= package_transmit_success
        self.env.energy_state -= package_transmit_success * e_t
        self.success_package_num += package_transmit_success


    def simulate(self):
        for i in range(1, self.time):
            self.perform_action(i)

    def perform_action(self, time):
        if self.can_transmit():
            self.transmit_data()
        else:
            self.harvest_energy()

        if time % 1000 == 0:
            print("Average throughput at " + str(time) + " is " + str(self.success_package_num / time) + ", package lost = " + str(self.package_lost))

        # data arrival
        data_arrive_l = poisson.rvs(mu=arrival_rate, size=1)
        data_arrive = data_arrive_l[0]
        self.env.data_state += data_arrive
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
