from logging import raiseExceptions
import torch
import pdb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import sys
import os

max_punish = 1e12

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class subagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(subagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 1)
        )

        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        single_state_hidden = self.fc1(single_state)
        trend_state_hidden = self.fc2(trend_state)
        c = action_hidden + trend_state_hidden
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(single_state_hidden), shift, scale)
        value = self.value(x)
        advantage = self.advantage(x)
        
        return value + advantage - advantage.mean()



class hyperagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(hyperagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1 + state_dim_2, hidden_dim)
        self.fc2 = nn.Linear(2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1)
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                class_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        c = self.fc2(class_state)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        weight = self.net(x)
        
        return weight

    def encode(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        return x


def calculate_q(w, qs):
    q_tensor = torch.stack(qs)
    q_tensor = q_tensor.permute(1, 0, 2)
    weights_reshaped = w.view(-1, 1, 6)
    combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
    
    return combined_q
