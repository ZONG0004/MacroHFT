import torch
import numpy as np
from collections import deque, namedtuple
import random
import pdb


def get_ada(ada,decay_freq=2,ada_counter=0, decay_coffient=0.5):
    if ada_counter % decay_freq==1:
        ada = decay_coffient*ada
    return ada


def get_epsilon( epsilon,max_epsilon=1, epsilon_counter=0, decay_freq=2,decay_coffient=0.5):
    if epsilon_counter%decay_freq == 1:
        epsilon =epsilon+(max_epsilon-epsilon)*decay_coffient
    return epsilon

class LinearDecaySchedule(object):
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, t):
        return max(self.end_epsilon, self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (t / self.decay_length))
