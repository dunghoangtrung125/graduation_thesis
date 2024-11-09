from enviroment import Environment
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Lambda, Add
from keras.models import Model
from keras import backend as K
from parameters import *

class DeepQLearningSimulation:

    def __init__(self):
        self.env = Environment()
        # load model
        self.model = tf.keras.models.load_model('result_model/dqn.keras')
