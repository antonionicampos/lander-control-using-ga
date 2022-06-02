<<<<<<< HEAD
import gym
import imageio

from agents import NN


env = gym.make("LunarLander-v2")

observation, info = env.reset(seed=42, return_info=True)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = NN(input_dim=input_dim, output_dim=output_dim, hidden_units=64)

action = agent(observation)

frames = []

done = False
total_reward = 0.0
while not done:
    env.render()
    # frame = env.render(mode='rgb_array')
    # frames.append(frame)
    action = agent(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(total_reward)
        observation, info = env.reset(return_info=True)

env.close()
=======
import gym
import imageio

from agents import NN


env = gym.make("LunarLander-v2")

observation, info = env.reset(seed=42, return_info=True)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = NN(input_dim=input_dim, output_dim=output_dim, hidden_units=64)

action = agent(observation)

frames = []

done = False
total_reward = 0.0
while not done:
    env.render()
    # frame = env.render(mode='rgb_array')
    # frames.append(frame)
    action = agent(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(total_reward)
        observation, info = env.reset(return_info=True)

env.close()
>>>>>>> 26a1559c357b19de545c8d61a9ff6ccf181d8600
# imageio.mimsave('movie.gif', frames)