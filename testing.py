"""
Testing the general RL algorithm on easy environment
"""
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import gym
import random
from collections import namedtuple
from torch.nn.init import kaiming_uniform_
import numpy as np


class DQN(nn.Module):
    """
    The policy network for appoximation of the Q function
    Model Parameters like in Mnih et al., 2015
    """

    def __init__(self, n_actions=4, feature_size=3136):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.feature_size = feature_size

        self.conv1 = nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer(object):
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
        if self.position >= self.capacity:
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


class Experience:
    """
    Class with slots to save an experience
    """
    __slots__ = ['state', 'action', 'reward', 'next_state', 'non_final']

    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.non_final = next_state is not None


def choose_action(state, epsilon, q_network, n_actions):
    """
    Choose an action given a state, epsilon and the q_network
    :param state:
    :param epsilon:
    :param q_network: the policy network
    :param n_actions:
    :return:
    """
    if random.random() < epsilon:
        with torch.no_grad():
            return torch.tensor(random.randrange(n_actions))
    else:
        with torch.no_grad():
            return torch.argmax(q_network.forward(state))


def transform_frame(frame):
    """
    Transform a frame to tensor of shape (batch_size, 1, 84, 84)
    :param frame:
    :return:
    """
    frame = T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.Grayscale(), T.ToTensor()])(frame)
    frame = frame / 255
    return frame

def game_step(env, action, n_steps=4):
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
        s, r, done, _ = env.step(action)
        frame = env.render(mode='rgb_array').transpose((2, 0, 1))
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

def extract_batch(batch, slot):
    if slot == 'action':
        return torch.tensor([x.action for x in batch])
    elif slot == 'state':
        return torch.cat([x.state for x in batch])
    elif slot == 'reward':
        return torch.tensor([x.reward for x in batch])
    elif slot == 'next_state':
        # return only non-final next states
        next_states = [x.next_state for x in batch]
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        return non_final_next_states
    elif slot == 'non_final':
        return torch.tensor([x.non_final for x in batch])
    else:
        raise Exception("Incorrect Experience slot specified")


def training_step(policy, target, memory, optimizer, criterion, batch_size=32, gamma=0.99, device="cuda"):
    """
    Calculate loss for one batch and perform optimization
    :param policy:
    :param memory:
    :return:
    """
    # check if enough memory has been aquired
    if len(memory) < batch_size:
        return

    # Sample batch from memory
    batch = memory.sample(batch_size)

    # extract state, action, reward, next_state
    action_batch = extract_batch(batch, 'action').to(device).unsqueeze(1)
    state_batch = extract_batch(batch, 'state').to(device)
    reward_batch = extract_batch(batch, 'reward').to(device)
    next_state_batch = extract_batch(batch, 'next_state').to(device)
    non_final_mask = extract_batch(batch, 'non_final')

    # q-values
    q_value = policy(state_batch)
    q_value = q_value.gather(1, action_batch).squeeze(1)

    # q-values for next state
    target_q_value_all = torch.zeros(batch_size, device=device)
    target_q_value = target(next_state_batch)
    _, max_idx = torch.max(target_q_value, dim=1)
    target_q_value = target_q_value.gather(1, max_idx.unsqueeze(1)).squeeze(1)
    target_q_value_all[non_final_mask] = target_q_value

    # expected q-value
    expected_q_value = reward_batch + (gamma * target_q_value_all)

    loss = criterion(q_value, expected_q_value)

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


"""
Main Section
"""

# === PARAMETERS ===#
batch_size = 8
max_screens = 1400000
memory_capacity = 30000
memory_init_size = 100
gamma = 0.99
target_update = 100
epsilon_start = 1
epsilon_end = 0.01
epsilon_steps = 2000
n_steps = 2
lr = 0.000625

# Create Breakout environment
env = gym.make('CartPole-v0')
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNNs
policy = DQN(n_actions=n_actions).cuda()
target = DQN(n_actions=n_actions).to(device)
target.load_state_dict(policy.state_dict())

# Loss
criterion = nn.SmoothL1Loss()

# Optimizer
optimizer = optim.Adam(policy.parameters(), lr=lr)

# Memory
memory = ReplayBuffer(memory_capacity)

epsilon = epsilon_start
epsilon_delta = (epsilon_start - epsilon_end) / epsilon_steps

screens = 0

while screens < max_screens:

    # reset env for new episode
    env.reset()
    # get initial state
    state, _, _ = game_step(env, env.action_space.sample(), n_steps=n_steps)
    complete_reward = 0

    for t in count():
        screens += 1

        # adjust epsilon
        epsilon = epsilon - epsilon_delta
        if epsilon <= epsilon_end:
            epsilon = epsilon_end

        # choose action based on current state
        action = choose_action(state.cuda(), epsilon, policy, n_actions).detach().cpu().numpy()

        # make one step
        next_state, reward, done = game_step(env, action, n_steps=n_steps)
        complete_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32)

        # save the current experience
        memory.push(Experience(state, action, reward, next_state))

        # update state variable
        state = next_state

        if screens > memory_init_size:
            # Perform one step of training on the policy network
            training_step(policy, target, memory, optimizer, criterion=criterion, batch_size=batch_size, gamma=gamma)

            # Update the target network after 10000 frames seen
            if screens % target_update == 0:
                target.load_state_dict(policy.state_dict())
                print("update target")

        if done:
            mem_len = len(memory.memory)
            print(f"Reward: {complete_reward}, Epsilon: {epsilon}, Memory: {mem_len}, Frames: {screens}")
            break