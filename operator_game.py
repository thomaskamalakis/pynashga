from matplotlib import pyplot as plt

from ga import population_group, population, counted, VERBOSITY_MEDIUM, VERBOSITY_HIGH, VERBOSITY_LOW
import numpy as np
from demand import demand_model as dm
from demand import profits as profs

# Game parameters
AREA_POPULATION = 10000
EXPANSION_COST = 0.2
COST_COEFF = 1
RESIDUAL_COEFF = 2
NO_PLAYERS = 2
NO_AREAS = 10
MAX_PRICE = 2.0
MIN_PRICE = 0.0

# Genetic algorithm parameters
VERBOSITY_LEVEL= VERBOSITY_LOW
NO_TOURNAMENTS = 1
MAX_FITNESS_EVALS = 1e5
MUTATION_FACTOR = 0.1
MIN_REL_FITNESS_RANGE = 1e-6
NO_CHROMOSOMES = 50
DEMAND_MODEL_TYPE = 'comp2'
MAX_CROSS_OVERS = 1e4
MAX_GENERATIONS = 1e4

class operator_game:

    def __init__(self,
                 no_players = NO_PLAYERS,
                 no_areas = NO_AREAS,
                 expansion_cost = EXPANSION_COST,
                 cost_coeff = COST_COEFF,
                 area_populations = AREA_POPULATION,
                 max_price = MAX_PRICE,
                 min_price = MIN_PRICE,
                 demand_model_type = DEMAND_MODEL_TYPE,
                 residual_coeff = RESIDUAL_COEFF,
                 verbosity_level = VERBOSITY_LEVEL,
                 no_tournaments = NO_TOURNAMENTS,
                 max_fitness_evals = MAX_FITNESS_EVALS,
                 mutation_factor = MUTATION_FACTOR,
                 min_rel_fitness_range = MIN_REL_FITNESS_RANGE,
                 no_chromosomes =  NO_CHROMOSOMES,
                 max_generations = MAX_GENERATIONS,
                 max_cross_overs = MAX_CROSS_OVERS
                 ):

        self.no_players = no_players
        self.no_areas = no_areas
        self.max_price = max_price
        self.min_price = min_price
        self.demand_model_type = demand_model_type
        self.residual_coeff = residual_coeff
        self.verbosity_level = verbosity_level
        self.no_tournaments = no_tournaments
        self.max_fitness_evals = max_fitness_evals
        self.mutation_factor = mutation_factor
        self.min_rel_fitness_range = min_rel_fitness_range
        self.no_chromosomes = no_chromosomes 
        self.max_generations = max_generations
        self.max_cross_overs = max_cross_overs

        if not isinstance( area_populations , np.ndarray):
           area_populations = area_populations * np.ones( self.no_areas )

        if not isinstance( expansion_cost, np.ndarray ):
           expansion_cost = expansion_cost * np.ones( [self.no_areas, self.no_players] )

        if not isinstance( cost_coeff, np.ndarray ):
           cost_coeff = cost_coeff * np.ones( [self.no_areas] )

        if not isinstance( residual_coeff, np.ndarray ):
           residual_coeff = residual_coeff * np.ones( [self.no_areas] )

        self.area_populations = area_populations
        self.cost_coeff = cost_coeff
        self.residual_coeff = residual_coeff
        self.expansion_cost = expansion_cost

        self.N = (self.no_areas + 1) * self.no_players
        self.mins = np.zeros( self.N )
        self.maxs = np.ones( self.N )

        i = np.arange(0, self.N, self.no_areas + 1)
        self.maxs[i] = max_price
        self.mins[i] = min_price
        self.variable_types = 'C' + 'I' * self.no_areas

        self.pg = population_group(
                         fitness_functions = self.fitness_functions,
                         variable_types = self.variable_types,
                         mins = self.mins,
                         maxs = self.maxs,
                         verbose_level = self.verbosity_level,
                         no_tournaments = self.no_tournaments,
                         max_fitness_evals = self.max_fitness_evals,
                         mutation_factors = self.mutation_factor,
                         no_populations = self.no_players,
                         min_rel_fitness_range = self.min_rel_fitness_range,
                         no_chromosomes = self.no_chromosomes,
                         max_generations = self.max_generations,
                         )

    def convert_to_vars(self, v):
        nvars = self.no_areas + 1
        c = 0

        p = np.zeros(self.no_players)
        b = np.zeros( [self.no_areas, self.no_players] )

        for j in range(self.no_players):
            vv = v[c: c + nvars]
            p[ j ] = vv[0]
            b[: , j] = vv[ 1: ]
            c += nvars

        return p, b

    def demand_model_comp1(self, v):
        n = np.zeros([self.no_areas, self.no_players])
        p, b = self.convert_to_vars(v)

        for i in range(self.no_areas):
            for j in range(self.no_players):
                n[i, j] = b[i, j] * self.area_populations[i] * np.exp( -self.cost_coeff[i] * p[j])

            if np.sum(n[i , :]) > self.area_populations[i]:
                n[i , :] = n[i , :] * self.area_populations[i] / np.sum(n[i , :])

        return n


    def demand_model_comp2(self, v):

        n = np.zeros([self.no_areas, self.no_players])
        p, b = self.convert_to_vars(v)

        for i in range(self.no_areas):

            # find the active users in each area for each player
            ni = b[i, :] * self.area_populations[i] * np.exp( -self.cost_coeff[i] * p )

            # maximum number of users for a player
            nmax = ni.max()
            imax = ni.argmax()
            pmin = p[imax]

            ni = b[i, :] * nmax * np.exp( -self.residual_coeff[i] * (p-pmin) )
            ntot = np.sum(ni)

            if ntot != 0:
                ni = nmax / ntot * ni
            else:
                ni = np.zeros( self.no_players )
            n[i, :] = ni
        return n

    def demand_model(self, v):

        if self.demand_model_type == 'comp1':
            return self.demand_model_comp1(v)

        elif self.demand_model_type == 'comp2':
            return self.demand_model_comp2(v)

        elif self.demand_model_type == 'comp3':
            return self.demand_model_comp3(v)


    def profits(self, v):
        p, b = self.convert_to_vars(v)

        profit = np.zeros( self.no_players )
        n = self.demand_model(v)

        for i in range(self.no_areas):
            for j in range(self.no_players):
                profit[j] += b[i,j] * (p[j] * n[i, j] - self.expansion_cost[i,j] * n[i, j])

        return profit

    @counted
    def fitness_functions(self, i, v):
        return self.profits(v)[i]


    def simulate(self):
        self.pg.simulate()

    def current_subscribers(self):
        v = self.pg.array_values()
        return self.demand_model(v)

    def current_profits(self):
        v = self.pg.array_values()
        return self.profits(v)

    def sensitivity(self, i, param_no, range_param):

        v0 = self.pg.array_values()
        e = np.zeros( v0.size )
        e[param_no] = 1
        r = np.zeros( range_param.size )

        for indx, dv in enumerate(range_param):
            v = v0 + e * dv
            r[indx] = self.profits(v)[i]

        return r

    def sensitivity_two_params(self, i, param_nos, ranges_params):

        x_range = ranges_params[0]
        ix = param_nos[0]
        y_range = ranges_params[1]
        iy = param_nos[1]
        v0 = self.pg.array_values()

        f = np.zeros([x_range.size, y_range.size])
        for indx, x in enumerate(x_range):
            for indy, y in enumerate(y_range):
               dv = np.zeros( v0.size )
               dv[ix] = x
               dv[iy] = y
               v = v0 + dv
               f[indx, indy] = self.profits(v)[i]
        return f
    
    def optimal_history_vars_list(self):
        optimal_vars = []
        for i, v in enumerate(self.pg.history_best):
            c_vars = []
            for vp in v:
               c_vars.append(vp.values())
            optimal_vars.append(c_vars)
            
        return optimal_vars
    
    def optimal_history_vars_arrays(self):
        optimal_vars_list = self.optimal_history_vars_list()
        optimal_vars = []
        for vl in optimal_vars_list:
            l = np.array([])
            for ve in vl:
                l = np.concatenate((l,ve))
            optimal_vars.append(l)
        return optimal_vars            

    def subscribers_history(self):
        opt_v = self.optimal_history_vars_arrays()
        subscribers = []
        for v in opt_v:
            n = self.demand_model(v)
            subscribers.append(n.reshape(-1))
            
        return subscribers
            
    
            

