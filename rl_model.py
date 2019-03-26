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
        self.n_actions = 4

        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(4480, 64)
        self.fc2 = nn.Linear(64, self.n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def choose_action(self, state, epsilon=0.8):
        """
        Choose which action to take
        :param sate:
        :param epsilon:
        :return:
        """
        if random.random() < epsilon:
            return torch.tensor(np.random.choice([0, 1, 2, 3]))
        else:
            with torch.no_grad():
                return torch.argmax(self.forward(state))

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
transform_screen = T.Compose([T.ToPILImage(), T.Resize(100),T.Grayscale(), T.ToTensor()])

def get_screen(environment):
    """
    Render and transform a screen given an environment
    :param environment:
    :return:
    """
    screen = environment.render(mode='rgb_array')
    screen = transform_screen(screen)
    screen = screen / 255 # normalization
    screen = screen.unsqueeze(0)
    return screen

def transform_frame(frame):
    """
    Transform a frame
    :param frame:
    :return:
    """
    frame = transform_screen(frame)
    frame = frame / 255
    return frame

def game_step(env, action, n_steps=3):
    """
    Play one step of the game
    :param env:
    :param action:
    :n_steps: number of frames to play
    :return:
    """
    frames = []
    reward = 0
    last_state = False
    for i in range(n_steps):
        frame, r, done, _ = env.step(action)
        if done:
            last_state = True
            reward += r
        else:
            frames.append(transform_frame(frame))
            reward += r
    if last_state:
        state = None
    else:
        state = torch.stack(frames, 0)
        state = state.transpose(0, 1)

    return (state, reward, last_state)

# Named tuple to store experiences in
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))


def training_step(policy, target, memory, optimizer, criterion, batch_size=32, gamma=0.9):
    """
    Perform optimization
    :param policy:
    :param memory:
    :return:
    """
    # check if enough memory has been aquired
    if len(memory) < batch_size:
        return

    batch = memory.sample(batch_size)
    batch = Experience(*zip(*batch))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)

    # Compute Policy values
    state_action_values = policy(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)

    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = reward_batch + gamma * next_state_values
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    loss = criterion(state_action_values, expected_state_action_values)

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

