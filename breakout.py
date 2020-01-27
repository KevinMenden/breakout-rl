"""
Learning how to play Breakout by Reinforcement Learning

"""
from itertools import count
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import gym
import random
import numpy as np
from collections import namedtuple
import math



"""
Reinforcement Learning model and functions necessary for implementing the algorithm
"""
class DQN(nn.Module):
    """
    The policy network for appoximation of the Q function
    Model Parameters like in Mnih et al., 2015
    """

    def __init__(self, n_actions=4):
        super(DQN, self).__init__()
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def choose_action(self, state, epsilon):
        """
        Choose which action to take
        :param state:
        :param epsilon:
        :return:
        """
        if random.random() < epsilon:
            with torch.no_grad():
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


# Transform input RGB image (210, 160, 3) to Grayscale Tensor of shape (bs, 1, 84, 84)  like in original
# publication
transform_screen = T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.Grayscale(), T.ToTensor()])


# def get_screen(environment):
#     """
#     Render and transform a screen given an environment
#     :param environment:
#     :return:
#     """
#     screen = environment.render(mode='rgb_array')
#     screen = transform_screen(screen)
#     screen = screen / 255 # normalization
#     screen = screen.unsqueeze(0)
#     return screen

def transform_frame(frame):
    """
    Transform a frame
    :param frame:
    :return:
    """
    frame = transform_screen(frame)
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


def training_step(policy, target, memory, optimizer, criterion, batch_size=32, gamma=0.9, device="cuda"):
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

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    non_final_next_states = non_final_next_states.to(device)

    state_batch = torch.cat(batch.state)
    state_batch = state_batch.to(device)
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    action_batch = action_batch.to(device)
    reward_batch = torch.cat(batch.reward)
    reward_batch = reward_batch.to(device)

    # Compute Policy values
    state_action_values = policy(state_batch).to(device)
    state_action_values = state_action_values.gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach().to(device)
    expected_state_action_values = reward_batch + gamma * next_state_values
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    loss = criterion(state_action_values, expected_state_action_values)

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


"""
Main Section
Running the algorithm and the game
"""

#=== PARAMETERS ===#
batch_size = 32
episodes = 100000
memory_capacity = 120000
memory_init_size = 20000
gamma = 0.9
target_update = 10
epsilon_start = 1
epsilon_end = 0.1
epsilon_steps = 1000000
n_steps = 4
lr = 0.00025
n_actions = 4
#==================#

# Create Breakout environment
env = gym.make('Breakout-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create networks
policy = DQN(n_actions=n_actions).to(device)
target = DQN(n_actions=n_actions).to(device)
target.load_state_dict(policy.state_dict())

# Define Loss
loss = nn.SmoothL1Loss()

# Create Optimizer
optimizer = optim.Adam(policy.parameters(), lr=lr)

# Create Memory
memory = Replaymemory(memory_capacity)

state_counter = 0
epsilon = epsilon_start

# Fill the memory up til memory_init_size
#for _ in range(memory_init_size):


# Play the game
for ep in range(episodes):

    env.reset() # reset environment
    # get initial state
    action = env.action_space.sample()
    state, _, _ = game_step(env, action, n_steps=n_steps)
    complete_reward = 0
    # play one episode
    for t in count():
        state_counter += 1

        if t % 2:
            env.render()

        # adjust epsilon
        epsilon = epsilon - (epsilon_start - epsilon_end)/epsilon_steps
        if epsilon < epsilon_end:
            epsilon = epsilon_end

        # choose an action based on the current state
        action = policy.choose_action(state.cuda(), epsilon=epsilon).data.cpu()

        # make one step with the action
        next_state, reward, done = game_step(env, action, n_steps=n_steps)
        complete_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32)

        # save the current experience
        memory.push(Experience(state, action, reward, next_state))

        # update state variable
        state = next_state

        if state_counter > memory_init_size:
            # Perform one step of training on the policy network
            training_step(policy, target, memory, optimizer, criterion=loss, batch_size=batch_size, gamma=gamma)

            if ep % target_update == 0:
                target.load_state_dict(policy.state_dict())

        if done:
            mem_len = len(memory.memory)
            print(f"Episode: {ep}, Reward: {complete_reward}, Epsilon: {epsilon}, Memory: {mem_len}")
            #print(epsilon)
            break


env.close()





