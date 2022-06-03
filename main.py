import imageio
import gym
import numpy as np
import os
import sys

from policies import NN


class EvolutionStrategy:
    
    def __init__(self, fitness_func, dim=2, pop_size=100):
        self.pop_size = pop_size
        self.dim = dim
        self.parents_size = self.pop_size // 2
        self.sigma_min = 0.00001
        self.tau1 = 1 / np.sqrt(2 * self.dim)
        self.tau2 = 1 / np.sqrt(2 * np.sqrt(self.dim))
        self.fitness = lambda x: -fitness_func(x[:self.dim])
        
        # Initializing population
        genes = np.random.uniform(-30., 30., size=(self.dim, pop_size))
        strat = np.random.uniform(10., 1., size=(self.dim, pop_size))
        self.pop = np.r_[genes, strat]
        
    def parents_selection(self):
        parents_indexes = np.random.choice(self.pop_size, self.parents_size, replace=False)
        parents = self.pop[:, parents_indexes]
        return parents
    
    def crossover(self, parents, lmbd=200):
        children = np.empty((2*self.dim, lmbd))
        for i in range(lmbd):
            parents_indexes = np.random.choice(parents.shape[1], 2, replace=False)
            genes = parents[:self.dim, parents_indexes]
            strat = parents[self.dim:, parents_indexes]
            
            # Discrete crossover for individual / Intermediate crossover for strategies
            # For individual
            crossover_indexes = np.random.choice(2, self.dim)
            child_genes = np.choose(crossover_indexes, genes.T).reshape(-1, 1)
            
            # For strategy
            child_strats = (strat[:, [0]] + strat[:, [1]]) / 2
            
            children[:, [i]] = np.r_[child_genes, child_strats]
            
        return children
    
    def mutation(self, children):
        genes = children[:self.dim, :]
        strat = children[self.dim:, :]
        
        # Uncorrelated mutation
        new_strat = strat * np.exp(self.tau1 * np.random.normal(size=(strat.shape[1]))) * np.exp(self.tau2 * np.random.normal(size=strat.shape))
        new_strat = np.where(new_strat > self.sigma_min, new_strat, self.sigma_min)
        new_genes = genes + new_strat * np.random.normal(size=genes.shape)
        
        return np.r_[new_genes, new_strat]
        
    def run(self, n_gens=10, sigma_min=0.00001):
        self.sigma_min = sigma_min
        self.populations = [self.pop]
        for gen in range(int(n_gens)):
            parents = self.parents_selection()
            children = self.crossover(parents)
            mutant_children = self.mutation(children)
            gen_fitness = self.fitness(mutant_children[:self.dim, :])
            
            best_indices = np.argsort(gen_fitness)
            best_fitnesses = gen_fitness[best_indices][-self.pop_size:]
            self.pop = mutant_children[:, best_indices[-self.pop_size:]]
            
            self.populations.append(self.pop)
            sys.stdout.write("\rGeneration %i" % (gen+1))
            sys.stdout.flush()
        print(f'\ngen {gen+1}, best fitnesses: {best_fitnesses[-5:]}')


if __name__ == "__main__":
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