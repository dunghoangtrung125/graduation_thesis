import numpy as np
import random
from scipy.stats import poisson
from parameters import *
from action import Action
from enum import Enum

class RealEnvironment:

    def __init__(self):
        self.data_state = []
        self.energy_store = []
        self.power = 0

        self.power_J = [0, 5, 15, 20]
        self.power_x = [0.1, 0.54, 0.18, 0.18]

    
    def perform_action(self):
        # data arrival
        data_arrive_l = poisson.rvs(mu=arrival_rate, size=1)
        data_arrive = data_arrive_l[0]
        self.data_state += data_arrive
        if self.data_state > d_queue_size:
            self.data_state = d_queue_size

        # jammer state
        self.power = np.random.choice(self.power_J, p=self.power_x)
        print(self.power)
    
        
