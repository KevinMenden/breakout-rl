#import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from pathlib import Path
import sys
import warnings
import timeit
from torch.utils import tensorboard

env = make_env('Pong-v0')
best_score = -np.inf
load_checkpoint = False
n_games = 1000
do_training = True

ckpt_dir = Path("C:/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/models")
figure_dir = Path("C:/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/plots")
log_dir = Path("C:/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/plots")

# for WSL
#ckpt_dir = Path("/mnt/c/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/models")
#figure_dir = Path("/mnt/c/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/plots")
#log_dir = Path("/mnt/c/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/plots")

writer = tensorboard.SummaryWriter(log_dir=log_dir)

agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                 input_dims=(env.observation_space.shape),
                 n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                 batch_size=32, replace=1000, eps_dec=1e-5,
                 chkpt_dir=ckpt_dir, algo='DQNAgent',
                 env_name='Pong-v0' )

if load_checkpoint:
    agent.load_models()

figure_file = figure_dir / "test_game.png"

n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_games):
    start = timeit.default_timer()
    done = False
    observation = env.reset()

    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward

        env.render()

        if do_training:
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            agent.learn()
        observation = observation_
        n_steps += 1
    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-50:])
    end = timeit.default_timer()
    game_time = end - start
    writer.add_scalar("Score", score, n_steps)
    writer.add_scalar("Epsilon", agent.epsilon, n_steps)
    
    print('episode: ', i,'score: ', score,
          ' average score %.1f' % avg_score, 'time %.2f' % game_time,
          'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    if avg_score > best_score:
        if do_training:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.epsilon)
    
    
writer.close()
x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores, eps_history, figure_file)