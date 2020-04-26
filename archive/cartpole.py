import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from archive.utils import plot_learning_curve

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


class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, epsilon=1.0, eps_dec=0.99999, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = DNN(n_features=input_dims, lr=self.lr, n_actions=self.n_actions)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()

        states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        states_ = torch.tensor(state_, dtype=torch.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = reward + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


# Main loop
env = gym.make("CartPole-v1")
n_games = 10000
scores = []
eps_history = []
agent = Agent(input_dims=env.observation_space.shape[0], n_actions=env.action_space.n, lr=0.0001)

for i in range(n_games):
    score = 0
    done = False
    obs = env.reset()

    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        score += reward
        agent.learn(obs, action, reward, obs_)

        # set old state to new state
        obs = obs_

    scores.append(score)
    eps_history.append(agent.epsilon)

    if i % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i}, score {avg_score}, epsilon {agent.epsilon}")


filename = 'cartpole_naive_dqn.png'
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, eps_history, filename)


