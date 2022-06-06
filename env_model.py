import imageio
import gym
import numpy as np
import os
import time

from policies import NN

results = np.load('results_gen1200.npz')
pop = results['pop']

env = gym.make("LunarLander-v2")

observation, info = env.reset(seed=np.random.randint(0, 2**16-1), return_info=True)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy = NN(input_dim=input_dim, output_dim=output_dim, hidden_units=16)
dim = policy.params_size()
policy.set_weights(pop[:dim, 25])

action = policy(observation)

frames = []

done = False
frame = 0
total_reward = 0.0
while not done:
    print(frame)
    env.render()
    # frames.append(env.render(mode='rgb_array'))
    action = policy(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward
    frame += 1
    if done:
        print(total_reward)

env.close()
# imageio.mimsave(os.path.join('.', 'movie.gif'), frames, format='GIF', fps=60)