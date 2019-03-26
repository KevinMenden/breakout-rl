"""
Learning how to play Breakout by Reinforcement Learning

"""
from rl_model import *
from itertools import count
import math

#=== PARAMETERS ===#
batch_size = 32
episodes = 10000
memory_capacity = 50000
gamma = 0.9
target_update = 5
epsilon_start = 0.9
epsilon_end = 0.1
epsilon_steps = 10000
n_steps=4
lr = 0.0001
n_actions=4
#==================#

# Create Breakout environment
env = gym.make('Breakout-v0')

# Create networks
policy = DQN(n_actions=n_actions)
target = DQN(n_actions=n_actions)
target.load_state_dict(policy.state_dict())


# Define Loss
loss = nn.MSELoss()

# Create Optimizer
optimizer = optim.Adam(policy.parameters(), lr=lr)

# Create Memory
memory = Replaymemory(memory_capacity)

state_counter = 0
epsilon = epsilon_start
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
        #epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.*state_counter / 10000)
        epsilon = epsilon - (epsilon_start - epsilon_end)/epsilon_steps

        # choose an action based on the current state
        action = policy.choose_action(state, epsilon=epsilon)

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
            #print("Episode: " + str(ep) + " , Reward: " + str(complete_reward) + ", Esplicon {.2f")
            print(f"Episode: {ep}, Reward: {complete_reward}, Epsilon: {epsilon}")
            #print(epsilon)
            break


env.close()





