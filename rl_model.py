"""
Reinforcement Learning model and functions necessary for implementing the algorithm
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

class DQN(nn.Module):
    """
    The policy network for appoximation of the Q function
    """

    def __init__(self):
        super(DQN, self).__init__()
        self.n_actions = 2

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, self.n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def choose_action(self, state, epsilon=1):
        """
        Choose which action to take
        :param sate:
        :param epsilon:
        :return:
        """
        if random.random() > epsilon:
            return torch.tensor([[random.random()]])
        else:
            with torch.no_grad():
                return self.forward(state)

class Replaymemory(object):
    """
    Object for saving the memory
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        """
        Saves an experience or just one timepoint
        :param experience:
        :return:
        """
        if self.position == self.capacity:
            self.position = 0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position += 1

    def sample(self, batch_size):
        """
        Take a random batch from the memory
        :param batch_size:
        :return:
        """
        batch = random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)


# Transform input RGB image (210, 160, 3) to Grayscale Tensor of shape (bs, 1, 65, 50)
transform_screen = T.Compose([T.ToPILImage(), T.Resize(50),T.Grayscale(), T.ToTensor()])

# Named tuple to store experiences in
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
