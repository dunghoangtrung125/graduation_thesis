import numpy as np
import random
from scipy.stats import poisson
from parameters import *

class Environment:
  def __init__(self):
    # state = 0: jammer is idle, state = 1: jammer attacks the system
    self.jammer_state = 0
    self.data_state = 0
    self.energy_state = 0

  def get_state(self):
    count = 0
    state = 0
    for jammer in range(0, 2):
      for data in range(0, d_queue_size + 1):
        for energy in range(0, e_queue_size + 1):
          if self.jammer_state == jammer and self.data_state == data and self.energy_state == energy:
            state = count
          count += 1
    return state

  def get_possible_action(self):
    list_actions = [0] # stay ilde
    if self.jammer_state == 0 and self.data_state > 0 and self.energy_state >= e_t:
      list_actions.append(1) # active transmit
    if self.jammer_state == 1:
      list_actions.append(2) # harvest energy
      if self.data_state > 0:
        list_actions.append(3) # backscattered
        if self.energy_state > e_t:
          list_actions.append(4) # RA1
          list_actions.append(5) # RA2
          list_actions.append(6) # RA3
    return list_actions

  def calculate_reward(self, action):
    reward = 0
    loss = 0
    if action == 0: # stay idle
      reward = 0
    elif action == 1: # active transmit
      reward = self.active_transmit(d_t)
    elif action == 2: # harvest energy
      reward = random.choices(e_hj_arr, weights=nu_p, k=1)[0]
    elif action == 3: # backscattered
      d_bj = random.choices(d_bj_arr, weights=nu_p, k=1)[0]

      if self.data_state >= b_dagger:
        max_rate = b_dagger
      else:
        max_rate = self.data_state

      if self.data_state > d_bj:
        reward = d_bj
      else:
        reward = self.data_state

      if max_rate > reward:
        loss = max_rate - reward

    elif action == 4: # RA1
      max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
      reward = self.active_transmit(dt_ra_arr[0])

      if dt_ra_arr[0] > max_ra:
        loss = reward
        reward = 0

    elif action == 5: # RA2
      max_ra = random.choices(dt_ra_arr, nu_p, k=1)[0]
      reward = self.active_transmit(dt_ra_arr[1])

      if dt_ra_arr[1] > max_ra:
        loss = reward
        reward = 0

    elif action == 6: #RA3
      reward = 0

    return reward, loss

  def active_transmit(self, maximum_transmit_packets):
    num_transmitted = 0
    if 0 < self.data_state < maximum_transmit_packets:
      if self.energy_state >= e_t * self.data_state:
        num_transmitted = self.data_state
      elif self.energy_state >= e_t:
        num_transmitted = self.energy_state // e_t
    else:
      if self.energy_state >= e_t * maximum_transmit_packets:
        num_transmitted = maximum_transmit_packets
      elif self.energy_state >= e_t:
        num_transmitted = self.energy_state // e_t
    return num_transmitted


  def perform_action(self, action):
    reward, loss = self.calculate_reward(action)
    if action == 1: # active transmit
      self.data_state -= reward
      self.energy_state -= reward * e_t
    elif action == 2: # harvest energy
      if self.energy_state < e_queue_size:
        self.energy_state += reward
      if self.energy_state > e_queue_size:
        self.energy_state = e_queue_size
      reward = 0
    elif action == 3:  # when perform backscatter, always backscatter 3 packages
      if self.data_state >= b_dagger:
        max_rate = b_dagger
      else:
        max_rate = self.data_state

      self.data_state -= max_rate
    elif action == 4 or action == 5:
      if reward > 0:
        self.data_state -= reward
        self.energy_state -= reward * e_t
      else:
        self.data_state -= loss
        self.energy_state -= loss * e_t

    # data arrival
    data_arrive_l = poisson.rvs(mu=arrival_rate, size=1)
    data_arrive = data_arrive_l[0]
    self.data_state += data_arrive
    if self.data_state > d_queue_size:
      self.data_state = d_queue_size

    # jammer state
    if self.jammer_state == 0:
      if np.random.random() <= 1 - nu:
        self.jammer_state = 1
    else:
      if np.random.random() <= nu:
        self.jammer_state = 0

    next_state = self.get_state_deep()
    return reward, next_state

  def get_state_deep(self):
    return np.array([self.jammer_state, self.data_state, self.energy_state])