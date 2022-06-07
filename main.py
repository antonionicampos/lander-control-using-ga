import gym
import imageio
import numpy as np
import os
import sys

from datetime import datetime
from policies import NN


class EvolutionStrategy:
    
    def __init__(self, pop_size=50, hidden_units=128):
        # Initializing gym environment
        env = gym.make("LunarLander-v2")

        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        env.close()
        self.hidden_units = hidden_units

        self.policy = NN(input_dim=self.input_dim, output_dim=self.output_dim, hidden_units=self.hidden_units)
        self.dim = self.policy.params_size()
        
        self.pop_size = pop_size
        self.lambda_ = 7 * self.pop_size    # High Selective Pressure 
        self.parents_size = self.pop_size // 2
        self.tau1 = 1 / np.sqrt(2 * self.dim)
        self.tau2 = 1 / np.sqrt(2 * np.sqrt(self.dim))
        
        # Initializing population
        weights1 = np.random.normal(0., 1., size=(self.hidden_units*self.input_dim, self.pop_size))
        weights2 = np.random.normal(0., 1, size=(self.output_dim*self.hidden_units, self.pop_size))
        bias1 = np.zeros((self.hidden_units, self.pop_size))
        bias2 = np.zeros((self.output_dim, self.pop_size))
        genes = np.r_[weights1, weights2, bias1, bias2]
        
        strat = np.random.uniform(1., 0.1, size=(self.dim, self.pop_size))
        self.pop = np.r_[genes, strat]
        
    def parents_selection(self):
        parents_indexes = np.random.choice(self.pop_size, self.parents_size, replace=False)
        parents = self.pop[:, parents_indexes]
        return parents
    
    def crossover(self, parents):
        children = np.empty((2*self.dim, self.lambda_))
        for i in range(self.lambda_):
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

    def fitness(self, population, gif={'create': False}):
        fitnesses = np.empty((population.shape[1]))
        
        start = datetime.now()
        for individual in np.arange(population.shape[1]):
            total_reward = 0.0
            
            # Set policy weights (genes)
            genes = population[:self.dim, individual]
            self.policy.set_weights(genes)
            
            env = gym.make("LunarLander-v2")
            observation, _ = env.reset(seed=np.random.randint(0, 2**16-1), return_info=True)

            if gif['create']: frames = []
            frame = 0
            done = False
            while not done:
                if gif['create']:
                    # env.render()
                    frames.append(env.render(mode='rgb_array'))
                action = self.policy(observation)
                observation, reward, done, _ = env.step(action)
                total_reward += reward

                frame += 1
                # if done: observation, _ = env.reset(return_info=True)
                if frame >= 200:
                    total_reward += -500. 
                    done = True
                    
            env.close()
            if gif['create']:
                imageio.mimsave(os.path.join('.', f'gen{gif["gen"]}_individual{individual}.gif'), 
                                frames, 
                                format='GIF', 
                                fps=60)
            fitnesses[individual] = total_reward
        elapsed_time = datetime.now() - start
        return fitnesses, elapsed_time
        
    def run(self, n_gens=10, sigma_min=0.00001):
        best_fitness_per_gen = np.empty(n_gens)
        avg_fitness_per_gen = np.empty(n_gens)

        filename = f'hidden_units_{self.hidden_units}_pop_size_{self.pop_size}'
        
        self.sigma_min = sigma_min
        
        for gen in range(int(n_gens)):
            parents = self.parents_selection()
            children = self.crossover(parents)
            mutant_children = self.mutation(children)
            
            gen_fitness, elapsed_time = self.fitness(mutant_children)
            
            best_indices = np.argsort(gen_fitness)[-self.pop_size:]
            
            best_fitnesses = gen_fitness[best_indices][-self.pop_size:]
            best_fitness_per_gen[gen] = best_fitnesses[-1]
            avg_fitness_per_gen[gen] = np.mean(best_fitnesses)
            
            self.pop = mutant_children[:, best_indices]

            if gen % 100 == 99: 
                np.savez(f'.\\results\\temp_gen_{gen+1}_{filename}', 
                         pop=self.pop, 
                         best_fitness_per_gen=best_fitness_per_gen, 
                         avg_fitness_per_gen=avg_fitness_per_gen)
            
            sys.stdout.write("\rGeneration {}, Elapsed Time {}".format(gen+1, elapsed_time))
            sys.stdout.flush()
            
        print(f'\ngen {gen+1}, best fitnesses: {best_fitnesses[-5:]}')
        return self.pop, best_fitness_per_gen, avg_fitness_per_gen


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_gens = 1000
    pop_size = 100
    hidden_units = 128
    
    es = EvolutionStrategy(pop_size=pop_size, hidden_units=hidden_units)

    start = datetime.now()
    pop, best_fitness_per_gen, avg_fitness_per_gen = es.run(n_gens=n_gens)
    print(f'Exec Time: {datetime.now() - start}')
    
    filename = f'gen_{n_gens}_hidden_units_{hidden_units}_pop_size_{pop_size}'

    np.savez(f'.\\results\\results_{filename}', 
             pop=pop, 
             best_fitness_per_gen=best_fitness_per_gen, 
             avg_fitness_per_gen=avg_fitness_per_gen)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n_gens), best_fitness_per_gen, label='best fitness')
    ax.plot(np.arange(n_gens), avg_fitness_per_gen, label='avg fitness')
    ax.set_ylabel('Best Fitness')
    ax.set_xlabel('Generation')
    ax.grid()

    plt.savefig(f'.\\plots\\plot_{filename}.png', dpi=500)
    plt.show()