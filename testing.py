import gym
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F



# create environment
env = gym.make('Pong-v0')

print(env.action_space)
print(env.action_space.sample())
