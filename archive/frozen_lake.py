import gym
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.Q = {}

    
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([a for a in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            action = np.argmax(actions)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    
    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        self.Q[(state, action)] += self.lr * (reward + self.gamma * self.Q[(state_, a_max)] - self.Q[(state, action)])

        self.decrement_epsilon()


class DNN(nn.Module):

    def __init__(self, n_features, lr, n_actions=4):
        super(DNN, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_features, 128)
        self.fc2 = nn.Linear(128, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





env = gym.make("FrozenLake-v0")
agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.9999995, n_actions=4, n_states=16)
agent.init_Q()
scores = []
win_pct_list = []
n_games = 500000

for i in range(n_games):
    done = False
    observation = env.reset()

    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, observation_)
        score += reward
        observation = observation_ 
    scores.append(score)

    if i % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)
        if i % 1000 == 0:
            print(f"episode {i} win_pct {win_pct}")

plt.plot(win_pct_list)
plt.show()