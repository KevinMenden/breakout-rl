import gym
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import random

# create environment
env = gym.make('Pong-v0')

print(env.action_space)
print(env.action_space.n)

random.randrange(4)
frames = np.empty((500000, 65, 65), dtype=np.uint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)