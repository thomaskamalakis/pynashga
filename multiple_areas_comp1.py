from matplotlib import pyplot as plt
import numpy as np
from operator_game import operator_game, VERBOSITY_LOW
import pickle
import random

np.random.seed(4)
random.seed(90)

def save_to_file(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)
        
def load_from_file(filename):
    with open(filename, 'rb') as f:
        v = pickle.load(f)
    
    return v

        
my_expansion_cost = np.zeros([10,2])
for l in range(10):
    my_expansion_cost[l,0] = 3 * l/10 + 0.2    
    my_expansion_cost[l,1] = 3 * l/10 + 0.2
    
# Game parameters
AREA_POPULATION = 1e6
EXPANSION_COST = my_expansion_cost
COST_COEFF = 1
RESIDUAL_COEFF = 2
NO_PLAYERS = 2
MAX_PRICE = 3.0
MIN_PRICE = 0.0
NO_AREAS = 10

# Genetic algorithm parameters
VERBOSITY_LEVEL= VERBOSITY_LOW
NO_TOURNAMENTS = 1
MAX_FITNESS_EVALS = 1e6
MUTATION_FACTOR = 0.1
MIN_REL_FITNESS_RANGE = 1e-3
NO_CHROMOSOMES = 50
DEMAND_MODEL_TYPE = 'comp1'
MAX_GENERATIONS = 1e4

g = operator_game(no_players = NO_PLAYERS,
                  no_areas = NO_AREAS,
                  expansion_cost = EXPANSION_COST,
                  cost_coeff = COST_COEFF,
                  area_populations = AREA_POPULATION,
                  max_price = MAX_PRICE,
                  min_price = MIN_PRICE,
                  demand_model_type = DEMAND_MODEL_TYPE,
                  residual_coeff = RESIDUAL_COEFF,
                  no_chromosomes = NO_CHROMOSOMES,
                  min_rel_fitness_range = MIN_REL_FITNESS_RANGE,
                  max_fitness_evals = MAX_FITNESS_EVALS,
                  max_generations = MAX_GENERATIONS)

g.simulate()
v = g.optimal_history_vars_arrays()
n = g.subscribers_history()
rang = np.arange(-0.25, 0.25, 0.01)
r1 = g.sensitivity(0, 0, rang )
r2 = g.sensitivity(1, 11, rang )

plt.close('all')
plt.figure(1)
plt.plot(g.pg.fitness_evolution())

plt.figure(2)
plt.plot(rang, r1, rang,r2)

fitness_evolution = g.pg.fitness_evolution()
save_to_file(fitness_evolution, 'results/a10_comp1_fe.pickle')
save_to_file(r1,'results/a10_comp1_sens1.pickle')
save_to_file(r2,'results/a10_comp1_sens2.pickle')
save_to_file(rang,'results/a10_comp1_rang.pickle')
save_to_file(v,'results/a10_comp1_optv.pickle')
save_to_file(n,'results/a10_comp1_subscr.pickle')


