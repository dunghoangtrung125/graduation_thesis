import numpy as np
import random
from scipy.stats import poisson
from parameters import *
from action import Action
from enum import Enum

class JammerState(Enum):
  IDLE = 0
  ATTACK = 1

class Environment:
  def __init__(self):
    self.jammer_state = JammerState.IDLE.value
    self.data_state = 0
    self.energy_state = 0
    self.total_packages_arrival = 0
    self.loss_packages = 0
    # Custom environment params (for draw graph)
    self.d_t = d_t
    self.nu = nu
    self.nu_p = nu_p

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
    list_actions = [Action.STAY_IDLE.value]
    if self.jammer_state == JammerState.IDLE.value and self.data_state > 0 and self.energy_state >= e_t:
      list_actions.append(Action.ACTIVE_TRANSMIT.value)

    if self.jammer_state == JammerState.ATTACK.value:
      list_actions.append(Action.HARVEST_ENERGY.value)
      if self.data_state > 0:
        list_actions.append(Action.BACKSCATTERED.value)
        if self.energy_state >= e_t:
          list_actions.append(Action.RA_1.value)
          list_actions.append(Action.RA_2.value)
          list_actions.append(Action.RA_3.value)
    return list_actions

  def calculate_reward(self, action):
    reward = 0
    loss = 0
    if action == Action.STAY_IDLE.value:
      reward = 0
    elif action == Action.ACTIVE_TRANSMIT.value:
      reward = self.active_transmit(self.d_t)

    elif action == Action.HARVEST_ENERGY.value:
      reward = random.choices(e_hj_arr, weights=self.nu_p, k=1)[0]

    elif action == Action.BACKSCATTERED.value:
      d_bj = random.choices(d_bj_arr, weights=self.nu_p, k=1)[0]

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

    elif action == Action.RA_1.value:
      max_ra = random.choices(dt_ra_arr, self.nu_p, k=1)[0]
      reward = self.active_transmit(dt_ra_arr[0])

      # Selected rate higher than success rate, so all packages will lost
      if dt_ra_arr[0] > max_ra:
        loss = reward
        reward = 0

    elif action == Action.RA_2.value:
      max_ra = random.choices(dt_ra_arr, self.nu_p, k=1)[0]
      reward = self.active_transmit(dt_ra_arr[1])

      # Selected rate higher than success rate, so all packages will lost
      if dt_ra_arr[1] > max_ra:
        loss = reward
        reward = 0

    elif action == Action.RA_3.value:
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
    self.loss_packages += loss

    if action == Action.ACTIVE_TRANSMIT.value:
      self.data_state -= reward
      self.energy_state -= reward * e_t
    elif action == Action.HARVEST_ENERGY.value:
      if self.energy_state < e_queue_size:
        self.energy_state += reward
      if self.energy_state > e_queue_size:
        self.energy_state = e_queue_size
      reward = 0
    elif action == Action.BACKSCATTERED.value:  # when perform backscatter, always backscatter 3 packages
      if self.data_state >= b_dagger:
        max_rate = b_dagger
      else:
        max_rate = self.data_state

      self.data_state -= max_rate
    elif action == Action.RA_1.value or action == Action.RA_2.value:
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
    self.total_packages_arrival += data_arrive
    if self.data_state > d_queue_size:
      # loss package discard when data queue full
      self.loss_packages += (self.data_state - d_queue_size)
      self.data_state = d_queue_size

    # jammer state
    if self.jammer_state == JammerState.IDLE.value:
      if np.random.random() <= 1 - self.nu:
        self.jammer_state = JammerState.ATTACK.value
    else:
      if np.random.random() <= self.nu:
        self.jammer_state = JammerState.IDLE.value

    next_state = self.get_state()
    return reward, next_state
  

  # Deep_Q_Learning
  def get_state_deep(self):
    return np.array([self.jammer_state, self.data_state, self.energy_state])
  
  def perform_action_deep(self, action):
    reward, loss = self.calculate_reward(action)
    self.loss_packages += loss

    if action == Action.ACTIVE_TRANSMIT.value:
      self.data_state -= reward
      self.energy_state -= reward * e_t
    elif action == Action.HARVEST_ENERGY.value:
      if self.energy_state < e_queue_size:
        self.energy_state += reward
      if self.energy_state > e_queue_size:
        self.energy_state = e_queue_size
      reward = 0
    elif action == Action.BACKSCATTERED.value:  # when perform backscatter, always backscatter 3 packages
      if self.data_state >= b_dagger:
        max_rate = b_dagger
      else:
        max_rate = self.data_state

      self.data_state -= max_rate
    elif action == Action.RA_1.value or action == Action.RA_2.value:
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
    self.total_packages_arrival += data_arrive
    if self.data_state > d_queue_size:
      # loss package discard when data queue full
      self.loss_packages += (self.data_state - d_queue_size)
      self.data_state = d_queue_size

    # jammer state
    if self.jammer_state == JammerState.IDLE.value:
      if np.random.random() <= 1 - self.nu:
        self.jammer_state = JammerState.ATTACK.value
    else:
      if np.random.random() <= self.nu:
        self.jammer_state = JammerState.IDLE.value

    next_state = self.get_state_deep()
    return reward, next_state
  

  ##############################################################
  def set_active_transmission_package_num(self, d_t):
    self.d_t = d_t


  def set_jammer_power(self, nu=nu, nu_p=nu_p):
    self.nu = nu
    self.nu_p = nu_p


