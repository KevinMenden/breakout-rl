import gym
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# create environment
env = gym.make('Breakout-v0')

print(env.action_space)
print(env.action_space.sample())

frames = np.empty((500000, 65, 65), dtype=np.uint)
