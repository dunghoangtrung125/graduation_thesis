from enviroment import Environment
from parameters import *
import random

class HarvestThenTransmitt:

    def __init__(self):
        self.env = Environment()
        self.batery = e_queue_size

    def can_transmit(self):
        return self.batery >= e_t
    
    def harvest_energy(self):
        harvest_energy = random.choices(e_hj_arr, weights=nu_p, k=1)[0]
        self.batery += harvest_energy

        if self.batery > e_queue_size:
            self.batery = e_queue_size

    def transmit_data(self):
        max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
