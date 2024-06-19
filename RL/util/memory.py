import torch
import torch.nn.functional as F

import pdb
from scipy.spatial.distance import cosine
import numpy as np
import pathlib
import sys
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")
from MacroHFT.model.net import *


def custom_kernel(h, hi):
    squared_distance = np.sum((h - hi) ** 2)
    return 1 / (squared_distance + 1e-3)

class episodicmemory():
    def __init__(self, capacity, k, state_dim, state_dim_2, hidden_dim, device):
        self.capacity = capacity
        self.k = k
        self.current_size = 0
        self.count = 0
        self.device = device
        self.buffer = {"single_state": np.zeros((self.capacity, state_dim)),
                        "trend_state": np.zeros((self.capacity, state_dim_2)),
                        "previous_action": np.zeros((self.capacity)),
                        "hidden_state": np.zeros((self.capacity, hidden_dim)),
                        "action": np.zeros((self.capacity)),
                        "q_value": np.zeros((self.capacity))
                       }

    def add(self, hidden_state, action, q_value, single_state, trend_state, previous_action):
        self.buffer["single_state"][self.count] = single_state
        self.buffer["trend_state"][self.count] = trend_state
        self.buffer["previous_action"][self.count] = previous_action
        self.buffer["hidden_state"][self.count] = hidden_state
        self.buffer["action"][self.count] = action
        self.buffer["q_value"][self.count] = q_value
        self.count = (self.count + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def query(self, query_hidden_state, action):
        if self.current_size != self.capacity:
            weighted_q_value = np.nan
        else:
            kernel_values = np.array([custom_kernel(query_hidden_state, hs) for hs in self.buffer["hidden_state"]])
            top_k_indices = np.argsort(kernel_values)[-self.k:]
            top_k_actions = self.buffer["action"][top_k_indices]
            top_k_q_values = self.buffer["q_value"][top_k_indices]
            mask = (top_k_actions == action).astype(float)
            weights = kernel_values[top_k_indices] / np.sum(kernel_values[top_k_indices])
            masked_weights = weights * mask
            normalized_weights = masked_weights / np.sum(masked_weights)
            weighted_q_value = np.dot(normalized_weights, top_k_q_values)

        return weighted_q_value

    def re_encode(self, model):
        batch_size = 512
        for i in range(0, self.capacity, batch_size):
            batch_end = min(i + batch_size, self.capacity)
            single_states_batch = torch.tensor(self.buffer["single_state"][i:batch_end], dtype=torch.float32).to(self.device)
            trend_states_batch = torch.tensor(self.buffer["trend_state"][i:batch_end], dtype=torch.float32).to(self.device)
            previous_actions_batch = torch.tensor(self.buffer["previous_action"][i:batch_end], dtype=torch.long).to(self.device)
            with torch.no_grad():
                updated_hidden_states = model.encode(single_states_batch, trend_states_batch, previous_actions_batch).cpu().numpy()
            self.buffer["hidden_state"][i:batch_end] = updated_hidden_states

