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
import matplotlib.pyplot as plt


class DQN(nn.Module):
    """
    The policy network for appoximation of the Q function
    Model Parameters like in Mnih et al., 2015
    """

    def __init__(self, n_actions=2):
        super(DQN, self).__init__()
        self.n_actions = n_actions

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

    def forward(self, x):
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


def game_step(env, action, n_steps=4):
    """
    Play one step of the game
    :param env:
    :param action:
    :n_steps: number of frames to play
    :return:
    """
    state, reward, done, _  = env.step(action)
    last_state = False
    if done:
        reward = 0
        last_state = True

    return (state, reward, last_state)


def extract_batch(batch, slot):
    if slot == 'action':
        return torch.tensor([int(x.action) for x in batch])
    elif slot == 'state':
        return torch.cat([torch.Tensor(x.state).unsqueeze(0) for x in batch])
    elif slot == 'reward':
        return torch.tensor([x.reward for x in batch])
    elif slot == 'next_state':
        # return only non-final next states
        next_states = [x.next_state for x in batch]
        non_final_next_states = torch.cat([torch.Tensor(s).unsqueeze(0) for s in next_states if s is not None])
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
    state_batch = extract_batch(batch, 'state').to(device)
    reward_batch = extract_batch(batch, 'reward').to(device)
    next_state_batch = extract_batch(batch, 'next_state').to(device)
    action_batch = extract_batch(batch, 'action').to(device)
    non_final_mask = extract_batch(batch, 'non_final')


    # q-values
    q_value = policy(state_batch)
    action_batch = action_batch.unsqueeze(1)
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
batch_size = 64
memory_capacity = 10000
memory_init_size = 100
gamma = 0.8
target_update = 100
epsilon_start = 1
epsilon_end = 0.05
epsilon_steps = 10000
n_steps = 1
lr = 0.001
eval_freq = 20
episodes = 4000

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
rewards = []

total_reward = 0
for ep in range(episodes):

    # reset env for new episode
    env.reset()
    # get initial state
    state, _, _ = game_step(env, env.action_space.sample(), n_steps=n_steps)
    complete_reward = 0
    for t in count():

        # adjust epsilon
        epsilon = epsilon - epsilon_delta
        if epsilon <= epsilon_end:
            epsilon = epsilon_end

        # choose action based on current state
        action = choose_action(torch.Tensor(state).cuda(), epsilon, policy, n_actions).detach().cpu().numpy()

        # make one step
        next_state, reward, done = game_step(env, action, n_steps=n_steps)
        complete_reward += reward
        total_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32)

        # save the current experience
        memory.push(Experience(state, action, reward, next_state))

        # update state variable
        state = next_state

        training_step(policy, target, memory, optimizer, criterion=criterion, batch_size=batch_size, gamma=gamma)

        if ep % target_update == 0:
            target.load_state_dict(policy.state_dict())

        if done and ep % eval_freq == 0:
            mem_len = len(memory.memory)
            avg_reward = total_reward / eval_freq
            rewards.append(avg_reward)
            print(f"Reward: {avg_reward}, Epsilon: {epsilon}, Memory: {mem_len}, Episodes: {ep}")
            total_reward = 0
            break
        if done:
            break


plt.plot(rewards)
plt.show()