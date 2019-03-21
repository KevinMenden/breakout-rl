"""
Learning how to play Breakout by Reinforcement Learning

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import gym
import random
import numpy as np
from collections import namedtuple
from rl_model import *

#=== PARAMETERS ===#
batch_size = 32
episodes = 50
memory_capacity = 10000
#==================#

# Create Breakout environment
env = gym.make('Breakout-v0')

# Create networks
policy = DQN()
target = DQN()
target.load_state_dict(policy.state_dict())

# Define Loss
loss = nn.MSELoss()

# Create Optimizer
optimizer = optim.Adam(policy.parameters())

# Create Memory
memory = Replaymemory(memory_capacity)


