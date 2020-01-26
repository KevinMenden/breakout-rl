"""
Learning how to play Breakout by Reinforcement Learning

"""
from rl_model import *
from itertools import count
from torch.autograd import Variable
import math

#=== PARAMETERS ===#
batch_size = 32
episodes = 10000
memory_capacity = 100000
memory_init_size = 50000
gamma = 0.9
target_update = 100
epsilon_start = 1
epsilon_end = 0.1
epsilon_steps = 100000
n_steps = 3
lr = 0.0001
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
loss = nn.MSELoss()

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





