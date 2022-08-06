from evolutionary_algorithms import EvolutionStrategy
from datetime import datetime

n_generations = 1
pop_size = 100
hidden_units = [64, 64]

es = EvolutionStrategy(pop_size=pop_size, hidden_units=hidden_units)

start = datetime.now()
pop, best_fitness_per_gen, avg_fitness_per_gen = es.run(n_gens=n_generations)
print(f'Exec Time: {datetime.now() - start}')