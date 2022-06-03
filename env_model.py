import imageio
import gym

from policies import NN


env = gym.make("LunarLander-v2")

observation, info = env.reset(seed=42, return_info=True)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy = NN(input_dim=input_dim, output_dim=output_dim, hidden_units=64)

action = policy(observation)

frames = []

done = False
total_reward = 0.0
while not done:
    env.render()
    frames.append(env.render(mode='rgb_array'))
    action = policy(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(total_reward)
        observation, info = env.reset(return_info=True)

env.close()
imageio.mimsave(os.path.join('.', 'movie.gif'), frames, format='GIF', fps=60)