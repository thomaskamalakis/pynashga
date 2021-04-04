import numpy as np
import random
from copy import deepcopy
import datetime

# Default values
# Constants
CONTINUOUS_REPRESENTATION = 'C'
INTEGER_REPRESENTATION = 'I'

# number of genes per chromosome and population size
NO_POPULATIONS = 2
NO_GENES = 10
VARIABLE_TYPES = CONTINUOUS_REPRESENTATION * NO_GENES

# Gene and fitness function print-out defaults
GENE_FORMAT = '{:.2f}'
FITNESS_FORMAT = '{:.2f}'

# Verbosity levels for the population class simulation - not used for Nash GA
VERBOSITY_LOW = 0
VERBOSITY_MEDIUM = 1
VERBOSITY_HIGH = 2
VERBOSITY_NONE = -1

# Default values for genetic algorithms
MUTATION_FACTOR = 0.03
SAVE_FILE = 'population.dat'
LOG_FILE = 'execution.log'
MAX_GENERATIONS = 1e4
NO_CHROMOSOMES = 10
MAX_CROSS_OVERS = 1000
TOURNAMENT_LEVEL = 0.5
REL_FITNESS_RANGE = 1e-6

NO_TOURNAMENTS = 1                  # number of tournaments per generation
POPULATION_REPLACEMENTS = 2         # max number of population members to be replaced
                                    # this is only for the population group class
MAX_FITNESS_EVALUATIONS = 10000     # Maximum number of fitness function evaluations
                                    # this is only for the population group class


# Some useful functions to be used later on


def listify( l, no_elements = 1, check_if_list = True ):
    """
    Returns a list with no_elements elements duplicate of l unless l is a list
    """
    if isinstance(l, list) and check_if_list:
        return l
    else:
        return [l] * no_elements

def is_list_of_lists(l):
    """
    Check if l is a list of lists
    """
    return all( [isinstance(el, list) for el in l] )

def time_stamp():
    """
    Current time
    """
    return datetime.datetime.now().strftime('%d-%m-%Y / %H:%M:%S')

def counted(f):
    """
    A decorator used to count the number of function evaluation
    """
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

@counted
def my_fitness_function(x):
    """
    Sample fitness function
    """
    return np.sum(x)


@counted
def my_fitness_functions(l):
    """
    Utility functions of M. Sefrioui and J. Perlaux, "Nash genetic algorithms:
    examples and applications," Proceedings of the 2000 Congress on Evolutionary Computation.
    CEC00 (Cat. No.00TH8512), La Jolla, CA, USA, 2000, pp. 509-516 vol.1,
    doi: 10.1109/CEC.2000.870339.
    """
    x = l[0]
    y = l[1]
    return [ -(x-1) ** 2.0 - (x-y) ** 2.0, -(y-3) ** 2.0 - (x-y) ** 2.0 ]


class gene:
    """
    The gene class representing a single variable in a chromosome
    """
    def __init__(self,
                 type = CONTINUOUS_REPRESENTATION, # type of variable (continuous or integer?)
                 init_to_random = None,            # assign random value?
                 g_value = None,                   # initial (untransformed) value for the original gene variables (in [0,1])
                 value = None,                     # actual gene value (transformed)
                 min_value = 0.0,                  # min value allowed for the gene value
                 max_value = 1.0):                 # max value allowed for the gene value

        self.type = type
        self.min_value = min_value
        self.max_value = max_value

        # initialize to random ?
        if init_to_random:
            self.randomize()

        # initialize directly from untransformed gene value?
        elif g_value is not None:
            self.set_g_value( g_value )

        # initialize directly from actual (transformed) value
        elif value is not None:
            self.set_value( value )

    def calc_value(self):
        """
        Calculate actual gene value using a slight variation of the transformation proposed in
        Haupt, Randy L. "Antenna design with a mixed integer genetic algorithm."
        IEEE Transactions on Antennas and Propagation 55.3 (2007): 577-582.
        """

        if self.type == CONTINUOUS_REPRESENTATION:
            v = (self.max_value - self.min_value) * self.g_value + self.min_value

        elif self.type == INTEGER_REPRESENTATION:
            # We use round here instead of floor proposed by the Haupt paper.
            v = np.round(
                (self.max_value - self.min_value) * self.g_value + self.min_value
                )

        self.value = v

        return v

    def randomize(self):
        """
        Initialize original value to random
        """
        self.g_value = np.random.rand()

    def get_g_value(self):
        """
        Get original gene value
        """
        return self.g_value

    def set_g_value(self, value):
        """
        Set original gene value
        """
        self.g_value = value

    def set_value(self, value):
        """
        Set actual (transformed) gene value explicitly
        """
        self.set_g_value(
          (value - self.min_value) / (self.max_value - self.min_value)
        )

class chromosome:
    """
    Chromose class used to model the population members
    """
    def __init__(self,
                 variable_type = VARIABLE_TYPES,            # default gene types
                 mins = [],                                 # lower bound for the genes
                 maxs = [],                                 # upper bound for the genes
                 gene_values = None,                        # untransformed gene values
                 values = None,                             # desired final gene values
                 init_to_random = None,                     # generate the gene values randomly
                 fitness_function = None,                   # fitness function
                 mutation_factor = MUTATION_FACTOR,         # mutation factor used
                 gene_format = GENE_FORMAT,                 # gene value print-out format
                 fitness_format = FITNESS_FORMAT,           # fitness value print-out format
                 mutation_type = 'continuous_mutatation',   # mutation type
                 cross_over_type = 'uniform_cross_over'     # cross over type
                 ):

        self.variable_type = variable_type
        self.gene_format = gene_format
        self.fitness_format = fitness_format
        self.mutation_type = mutation_type
        self.cross_over_type = cross_over_type

        if ( len(maxs) == 0 ) or ( len(maxs) == 0 ):
            mins = np.zeros( len(self.variable_type) )
            maxs = np.ones( len(self.variable_type) )

        self.mins = mins
        self.maxs = maxs
        self.mutation_factor = mutation_factor
        self.genes = []

        # Random genes
        if init_to_random:
            for i, v in enumerate( self.variable_type ):
                self.genes.append( gene(type = v,
                                        min_value = mins[i],
                                        max_value = maxs[i],
                                        init_to_random = True,
                                        ) )

        # Set genes directly from gene values
        elif gene_values is not None:
            for i, v in enumerate( self.variable_type ):
                self.genes.append( gene(type = v,
                                        min_value = mins[i],
                                        max_value = maxs[i],
                                        init_to_random = False,
                                        g_value = gene_values[i]) )

        # otherwise set values directly from desired values
        elif values is not None:
            for i, v in enumerate( self.variable_type ):
                self.genes.append( gene(type = v,
                                        min_value = mins[i],
                                        max_value = maxs[i],
                                        init_to_random = False,
                                        value = values[i]) )

        self.update_fitness_function( fitness_function )

    def has_fitness( self ):
        """
        Does the chromosome has a fitness value assigned to it?
        """
        return hasattr(self, 'fitness')

    def update_fitness_function( self, callable):
        """
        Update the fitness function callable of the chromosome
        """
        self.fitness_function = callable

    def set_g_values(self, gene_values ):
        """
        assign values for untransformed gene values
        """
        for i, g_value in enumerate(gene_values):
            self.genes[i].set_g_value(g_value)

    def size(self):
        """
        Chromosome size, i.e. number of genes
        """
        return len( self.variable_type )

    def __getitem__(self, i):
        """
        overload the index operator to return the actual (transformed) value of gene idea
        """
        return self.genes[i].calc_value()

    def values(self):
        """
        returns all the actual values of the genes of the chromosome
        """
        return np.array( [ self[i] for i in range(0, self.size()) ] )

    def g_values(self):
        """
        returns the initial (untransformed) values of the chromosome
        """
        return np.array( [ self.genes[i].get_g_value() for i in range(0, self.size()) ] )

    def set_values(self, vals):
        """
        Set the actual (transformed) gene values equal to vals
        """
        for i, g in enumerate(self.genes):
            g.set_value( vals[i] )

    def calc_fitness(self):
        """
        Calculate chromsome fitness (if fitness function callable is provided)
        """
        if hasattr(self, 'fitness_function'):
            if self.fitness_function is not None:
                self.fitness = self.fitness_function( self.values() )
                return self.fitness

    def calc_external_fitness(self, f):
        """
        Calculate chromosome fitness using some external function callable f
        """
        self.fitness = f( self.values() )
        return self.fitness

    """
    The following functions overload the comparison operators based on the fitness values
    """
    def __lt__(self, c):

        return self.fitness < c.fitness

    def __gt__(self, c):
        return self.fitness > c.fitness

    def __le__(self, c):
        return self.fitness <= c.fitness

    def __ge__(self, c):
        return self.fitness >= c.fitness

    def __eq__(self, c):
        return self.fitness == c.fitness

    def __str__(self):
        """
        String value of the chromosome comprises of all gene actual (transformed) values
        and the fitness function (if it exists)
        """
        gene_values = self.values()
        msg  = '[' + ','.join( [self.gene_format.format(x) for x in gene_values] ) + ']'
        if self.has_fitness():
            fitness = self.fitness
            msg += '->' + self.fitness_format.format(fitness)
        return msg

    def __repr__(self):
        """
        Representation of the chromosome comprises of all gene actual (transformed) values
        and the fitness function (if it exists)
        """
        return self.__str__()

    def uniform_cross_over(self, other_parent):

        """
        Cross over between two chromosomes, self and other_parent. We use the cross over process
        suggested in Haupt, Randy L. "Antenna design with a mixed integer genetic algorithm."
        IEEE Transactions on Antennas and Propagation 55.3 (2007): 577-582.
        """

        gene_values = []

        self_values = deepcopy( self.g_values() )
        other_parent_values = deepcopy( other_parent.g_values() )

        map = np.random.randint(0, high = 2, size = self.size() )

        # Randomly select genes from each parent to form the offspring
        for i, selection in enumerate(map):
            if selection == 0:
                gene_values.append( self_values[i] )
            else:
                gene_values.append( other_parent_values[i] )

        return chromosome(variable_type = self.variable_type,
                          mins = self.mins,
                          maxs = self.maxs,
                          gene_values = gene_values,
                          fitness_function = self.fitness_function,
                          mutation_factor = self.mutation_factor,
                          init_to_random = False
                          )

    def cross_over(self, other_parent):
        """
        Cross over operation
        """
        if self.cross_over_type == 'uniform_cross_over':
            return self.uniform_cross_over(other_parent)

    def __add__(self, c):
        """
        Addition is overloaded with cross over operation
        """
        return self.cross_over(c)

    def continuous_mutatation(self):
        """
        Mutation operation
        ar = self.mutation_factor suggested in Haupt, Randy L.
        "Antenna design with a mixed integer genetic algorithm.",
        IEEE Transactions on Antennas and Propagation 55.3 (2007): 577-582.
        """

        # Random gene mutipliers chosen randomly inside [-1,1]
        br = 1 - 2.0 * np.random.rand( self.size() )

        # gene values are the remainder of the original chromosomes augmented by
        # the gene mutations
        gene_values = self.mutation_factor * ( br * self.g_values() ) + self.g_values()
        gene_values = gene_values - np.floor(gene_values)

        self.set_g_values( gene_values )

    def mutate(self):
        """
        Mutate chromosome
        """
        if self.mutation_type == 'continuous_mutatation':
            self.continuous_mutatation()

class population:
    """
    population class used to store the chromosomes at each turn of the algorithm
    """

    def __init__(self,
                 no_chromosomes = NO_CHROMOSOMES,    # number of chromosomes in the population
                 variable_type = VARIABLE_TYPES,     # gene variable types
                 mins = [],                          # minimum value of the actual gene values
                 maxs = [],                          # maximum value of the actual gene values
                 fitness_function = None,            # fitness function for the population
                 mutation_factor = MUTATION_FACTOR,  # default mutation factor
                 seed = None,                        # random seed, None implies pseudo-random value for the seed
                 max_cross_overs = MAX_CROSS_OVERS,  # maximum number of cross-overs - NOT IMPLEMENTED YET!
                 save_filename = SAVE_FILE,          # filename to save population data - NOT IMPLEMENTED YET!
                 progress_filename = LOG_FILE,       # log file for the genetic algorithm execution - essentially
                                                     # duplicates print messages
                 verbose_level = VERBOSITY_LOW,      # verbosity level
                 gene_format = GENE_FORMAT,          # standard gene format for print-outs
                 fitness_format = FITNESS_FORMAT,    # standard fitness format for print-outs
                 init_to_random = True,              # initialize entire population to random genes
                 level = TOURNAMENT_LEVEL,           # fraction of the population participating in the tournament selection
                 max_generations = MAX_GENERATIONS,  # maximum number of generations allowed
                 calc_fitness = True,                # determines whether fitnesses will be calculated at the initialization
                                                     # of the population
                 rel_fitness_range = REL_FITNESS_RANGE, # if the relative fitness range is smaller than this the simulation will terminate
                 no_tournaments = NO_TOURNAMENTS     # number of tournaments per generation
                 ):

        self.mins = mins
        self.maxs = maxs
        self.no_chromosomes = no_chromosomes
        self.variable_type = variable_type
        self.fitness_function = fitness_function
        self.mutation_factor = mutation_factor
        self.seed = seed
        self.max_cross_overs = max_cross_overs
        self.verbose_level = verbose_level
        self.progress_filename = progress_filename
        self.init_to_random = init_to_random
        self.level = level
        self.generations = 0
        self.no_cross_overs = 0
        self.min_fitnesses = []
        self.max_fitnesses = []
        self.best_chromosomes = []
        self.max_generations = max_generations
        self.fitness_format = fitness_format
        self.gene_format = gene_format
        self.min_rel_fitness_range = rel_fitness_range
        self.no_tournaments = no_tournaments
        self.no_replacements = 0

        # Make sure progess file name is indeed required
        if verbose_level == VERBOSITY_NONE:
            self.progress_filename = None

        # If limits are not given, just assume [0, 1]
        if ( len(maxs) == 0 ) or ( len(maxs) == 0 ):
            mins = np.zeros( len(self.variable_type) )
            maxs = np.ones( len(self.variable_type) )

        self.mins = mins
        self.maxs = maxs

        # Determine random seed
        # self.set_seed()

        # Check to see if chromosomes should be initialized at random
        if self.init_to_random:
            self.report('Initializing random population.', VERBOSITY_MEDIUM)
            self.init_random_population(calc_fitness = calc_fitness)

        # Check if fitness needs to be evaluated at the initialization stage
        if calc_fitness:
            self.calc_fitnesses()

    def __getitem__(self, i):
        """
        Overload the index operation to return the correspding chromosomes at the index
        contained in i which can also be a list.
        """

        if isinstance(i, list):
            return [self.chromosomes[ j ] for j in i]

        return self.chromosomes[ i ]

    def size(self):
        """
        Size of the population
        """
        return len(self.chromosomes)

    def init_log_file(self):
        """
        Initialize log file
        """
        if self.progress_filename is not None:
            with open(self.progress_filename, 'w') as f:
                print('Nash GA execution log', file = f)

    def report(self, msg, verb_level):
        """
        Report something if population verbosity is higher or equal to verb_level
        """
        if verb_level <= self.verbose_level:
            tmsg = time_stamp() + ' > ' + str(msg)
            print(tmsg)
            if self.progress_filename is not None:
                with open(self.progress_filename, 'a') as f:
                    print(tmsg, file = f)

    def report_population(self):
        """
        Report current population state
        """

        # if verbosity is VERBOSITY_HIGH print out all chromosomes of the population
        self.report('Current pool:', VERBOSITY_HIGH)
        for chromosome in self.chromosomes:
            self.report(chromosome, VERBOSITY_HIGH)

        # if verbosity is higher or equal to VERBOSITY_MEDIUM report fittest chromosome
        # and fitness range
        self.report('Best chromosome so far:', VERBOSITY_LOW)
        self.report(self[0], VERBOSITY_LOW)
        if self.can_sort():
            self.report('Fitness range for population: ' + self.fitness_range_str() , VERBOSITY_MEDIUM )
            self.report('Fitness relative range for population: %6.3f' %self.rel_fitness_range() , VERBOSITY_MEDIUM )

        self.report('Number of cross-overs: %d' %self.no_cross_overs, VERBOSITY_MEDIUM)
        self.report('Number of generations: %d' %self.generations, VERBOSITY_MEDIUM)
        if hasattr(self.fitness_function, 'calls'):
            self.report('Number of fitness function evaluations: %d' %self.fitness_function.calls, VERBOSITY_MEDIUM)

    def can_sort(self):
        """
        Returns true if all chromosomes have fitnesses
        """
        return all( [ c.has_fitness() for c in self.chromosomes ] )

    def fitness_range_str(self):
        """
        returns a string containing the fitness range of the chromosomes
        """
        if self.can_sort():
            return self.fitness_format.format( self[0].fitness ) + ', ' + self.fitness_format.format( self[-1].fitness )
        else:
            return ''

    def calc_fitnesses(self):
        """
        Calculate the fitnesses of all chromosomes
        """
        for c in self.chromosomes:
            c.calc_fitness()

    def calc_external_fitnesses(self, f):
        """
        Calculate the fitnesses of all chromosomes using some external fitness callable
        """
        for c in self.chromosomes:
            c.calc_external_fitness(f)

    def init_random_population(self, calc_fitness = True):
        """
        Initialize population at random
        """
        self.chromosomes = [ chromosome( variable_type = self.variable_type,
                                         mins = self.mins,
                                         maxs = self.maxs,
                                         fitness_function = self.fitness_function,
                                         mutation_factor = self.mutation_factor,
                                         gene_format = self.gene_format,
                                         fitness_format = self.fitness_format,
                                         init_to_random = True,
                                         ) for i in range( self.no_chromosomes ) ]

        self.calc_fitnesses()
        self.sort_population()
        self.report_population()

    def fitnesses(self):
        """
        Return fitness of all chrosomes in the population
        """
        return [ c.fitness for c in self.chromosomes ]

    def rel_fitness_range(self):
        fitnesses = self.fitnesses()
        return np.abs( (np.max(fitnesses) - np.min(fitnesses)) / np.max(fitnesses) )

    def sort_population(self):
        """
        Sort population according to fitness in descending order, i.e. chromosome 0 is the
        chromosome with the strongest fitness and so on.
        """
        if self.can_sort():
            self.chromosomes.sort(reverse = True, key = lambda x: x.fitness )
            return True
        else:
            return False

    def set_seed(self, seed = None):
        """
        Set random seed used in simulations
        """
        if seed is None:
            seed = self.seed

        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.report('Setting seed to %s' %self.seed, VERBOSITY_HIGH )

    def tournament_selection(self):
        """
        tournament selection: returns the indices of chromosomes that will participate
        in the selection process.
        self.level is the percentage of strongest chromosomes participating in the tournament
        """
        indices = []
        i = 0
        no_participants = round( self.level * self.size() )

        self.report('Starting tournament selection.', VERBOSITY_MEDIUM)
        self.report('Number of chromosomes participating in the tournament: %s' %no_participants, VERBOSITY_HIGH)

        while i < 2:
            r = random.randint(0, no_participants)
            if r not in indices:
                indices.append(r)
                i += 1

        self.report('Selected indices: %s' %indices, VERBOSITY_HIGH)
        self.report('Selected chromosomes:', VERBOSITY_HIGH)
        for i in indices:
            self.report( self.chromosomes[i], VERBOSITY_HIGH )

        return [self.chromosomes[i] for i in indices]

    def replace(self, c):
        """
        replace the weakest chromosome with chromosome c if c is stronger.
        """

        fitnesses = self.fitnesses()

        self.report('Checking for elimination.', VERBOSITY_MEDIUM)

        # Replacement takes place only if chromosome c is stronger than the weakest
        # chromosome
        if c.fitness > np.min( fitnesses ):
            i = np.argmin( fitnesses )
            old_chromosome = deepcopy( self.chromosomes[i] )
            self.chromosomes[i] = deepcopy( c )
            self.report('Replaced chromosome:', VERBOSITY_HIGH)
            self.report(old_chromosome, VERBOSITY_HIGH)
            self.report('with chromosome:', VERBOSITY_HIGH)
            self.report(self.chromosomes[ i ], VERBOSITY_HIGH)

            # Do a sorting to make sure fitnesses are sorted for the final pool
            self.sort_population()
            return True
        else:
            self.report('No elimination carried out.', VERBOSITY_HIGH)
            return False
    def update_history(self):
        """
        Update simulation history
        """
        self.best_chromosomes.append( self[0] )
        self.max_fitnesses.append( self[0].fitness )
        self.min_fitnesses.append( self[-1].fitness )

    def next_generation(self, external_fitness = None):
        """
        Population is advanced one generation and is returned as a new population object.
        If external fitness function is provided then it is used to calculate the fitness of the offspring.
        No logging is done here
        """

        # We need to copy the population to make sure no weird by reference tricks are done here
        c = deepcopy(self)

        for t in range(self.no_tournaments):
            # Tournament selection
            [parent1, parent2] = c.tournament_selection()

            # Cross over
            offspring = parent1 + parent2

            # Mutation
            offspring.mutate()

            # Fitness calculation
            if external_fitness is None:
                offspring.calc_fitness()
            else:
                offspring.calc_external_fitness(external_fitness)

            # Update pool
            is_replaced = c.replace( offspring )
            if is_replaced:
                c.no_replacements += 1

            c.no_cross_overs += 1

        c.generations += 1

        # Update history
        c.update_history()

        return c

    def update_generation(self, external_fitness = None):
        """
        update the population by one generation. Unlike the next_generation method, population
        is replaced locally at the class.
        """

        self.report('', VERBOSITY_MEDIUM)
        self.report('Updating generation, iteration : %d ' %self.generations, VERBOSITY_LOW)

        for t in range(self.no_tournaments):

            self.report('Starting tournament %d' %t, VERBOSITY_MEDIUM)

            # Tournament selection
            [parent1, parent2] = self.tournament_selection()

            # Cross over
            offspring = parent1 + parent2
            self.no_cross_overs += 1
            self.report('Generating offspring', VERBOSITY_MEDIUM)
            self.report('Offspring:', VERBOSITY_HIGH)
            self.report(offspring, VERBOSITY_HIGH)

            # Mutation
            self.report('Mutating offspring', VERBOSITY_MEDIUM)
            offspring.mutate()

            if external_fitness is None:
                offspring.calc_fitness()
            else:
                offspring.calc_external_fitness(external_fitness)

            self.report('Mutated offspring:', VERBOSITY_HIGH)
            self.report(offspring, VERBOSITY_HIGH)

            # Update pool
            is_replaced = self.replace( offspring )

            if is_replaced:
                self.no_replacements += 1

        self.generations += 1

        # Report population
        self.report_population()

        # Update history
        self.update_history()

    def terminate(self):
        """
        Termination conditions
        """
        return (self.generations > self.max_generations) or \
               (self.no_cross_overs > self.max_cross_overs) or \
               (self.rel_fitness_range() < self.min_rel_fitness_range)

    def update_fitness_function(self, f ):
        """
        Update the fitness functions of all chromosomes
        """
        for c in self.chromosomes:
            c.update_fitness_function( f )

    def simulate(self):
        """
        Execute the genetic algorithm until a termination condition is met
        """
        self.init_log_file()
        self.report('Execution starting.', VERBOSITY_LOW)

        while not self.terminate():
            self.update_generation()

    def __str__(self):
        msg = ''
        for i, c in enumerate(self.chromosomes):
            msg += '%d. %s\n' %(i, c)
        return msg

    def __repr__(self):
        return self.__str__()

class population_group:
    """
    Class used for group of population. Each population has its own chromosome pool with possibly
    different gene variables and fitness function that contain the fittest chromosomes of the other
    populations obtained in the previous turn.
    The fitness functions have two arguments: a) the index of the population and b) a second argument containing:
    i) the variables placed on a numpy array of dimension equal to the sum of the number of variables
       on each population. So for two populations with 3 genes per chromosome we have a numpy array of size 6 or
    ii) the variables as a list of numpy arrays, one numpy array for the genes of each population. In the previous
       example we have a list of two numpy arrays, each of size 3
    """
    def __init__(self,

                 no_populations = NO_POPULATIONS,     # number of population groups
                 no_chromosomes = NO_CHROMOSOMES,     # number of chromosomes per population
                 variable_types = VARIABLE_TYPES,     # population variable types in string notation.
                                                      # Can be a list (if they are different for each group)
                 mins = None,                         # maximum and minimum values allowed for the gene values
                 maxs = None,                         # None implies [0, 1] range for all gene variables of all groups
                 fitness_functions = None,            # list of fitness functions used for each group. Can return a numpy array
                                                      # with one value for each group
                 mutation_factors = MUTATION_FACTOR,  # mutation factor for each group
                 gene_formats = GENE_FORMAT,          # string for gene print-out format
                 fitness_formats = FITNESS_FORMAT,    # string for fitness print-out format
                 calc_on_list = False,                # if true this implies that fitness functions are calculated on
                                                      # the argument on a list basis
                 verbose_level = VERBOSITY_LOW,       # verbose level
                 progress_filename = LOG_FILE,        # log file where reports are written
                 max_generations = MAX_GENERATIONS,   # max number of generations allowed for each population
                 max_fitness_evals = MAX_FITNESS_EVALUATIONS, # max number of fitness function evaluations allowed
                 min_rel_fitness_range = REL_FITNESS_RANGE, # Minimum relative fitness range required for all populations
                 no_tournaments = NO_TOURNAMENTS
                 ):

        self.no_populations = no_populations
        self.fitness_functions = fitness_functions
        self.fitness_formats = self.listify( fitness_formats )
        self.gene_formats = self.listify( gene_formats )
        self.mutation_factors = self.listify( mutation_factors )
        self.variable_types = self.listify( variable_types )
        self.no_chromosomes = self.listify( no_chromosomes )
        self.no_tournaments = self.listify( no_tournaments )
        self.populations = []
        self.calc_on_list = calc_on_list
        self.verbose_level = verbose_level
        self.progress_filename = progress_filename
        self.max_generations = max_generations
        self.max_fitness_evals = max_fitness_evals
        self.generations = 0
        self.history_best = []                                  # this is where the best chromosomes of each generations are kept
        self.min_rel_fitness_range = min_rel_fitness_range
        self.no_crossovers = 0

        if mins is None:
            mins = [[] for x in range(self.no_populations) ]

        if maxs is None:
            maxs = [[] for x in range(self.no_populations) ]

        if not is_list_of_lists(mins):
            self.mins = self.listify(mins, check_if_list = False)
        else:
            self.mins = mins

        if not is_list_of_lists(maxs):
            self.maxs = self.listify(maxs, check_if_list = False)
        else:
            self.maxs = maxs

        self.init_log_file()
        self.init_populations()
        self.init_fitnesses()
        self.choose_best_chromosomes()
        self.calc_fitnesses()

    def init_log_file(self):
        """
        Initialize log file
        """
        if self.progress_filename is not None:
            with open(self.progress_filename, 'w') as f:
                print('group GA execution log', file = f)

    def report(self, msg, verb_level, include_stamp = True):
        """
        Report something if population verbosity is higher or equal to verb_level
        """

        if verb_level <= self.verbose_level:
            if include_stamp:
               tmsg = time_stamp() + ' > ' + str(msg)
            else:
               tmsg = str(msg)
            print(tmsg)
            if self.progress_filename is not None:
                with open(self.progress_filename, 'a') as f:
                    print(tmsg, file = f)

    def listify(self, l, check_if_list = True):
        """
        Listify the variable l to a list of self.no_population elements
        This is used in the __init__ function to simplify the input variables
        """
        return listify(l, no_elements = self.no_populations, check_if_list = check_if_list)

    def init_populations(self):
        """
        Initialize the populations of the group
        """
        for i in range(self.no_populations):
            self.populations.append(
                population(
                    no_chromosomes = self.no_chromosomes[i],
                    variable_type = self.variable_types[i],
                    mins = self.mins[i],
                    maxs = self.maxs[i],
                    fitness_function = None,
                    mutation_factor = self.mutation_factors[i],
                    gene_format = self.gene_formats[i],
                    fitness_format = self.fitness_formats[i],
                    calc_fitness = False,
                    verbose_level = VERBOSITY_NONE,
                    no_tournaments = self.no_tournaments[i]
                )
            )

    def choose_best_chromosomes(self):
        """
        Identifies the best chromosomes of each population. This is required in the fitness
        function evaluations which must account for the genes of other populations as well.
        """
        self.best_chromosomes = [ x[0] for x in self.populations ]

    def init_fitnesses(self):
        """
        Initializes the fitness functions to -Inf. This is only used in the initialization stage
        and implies that at the initialization stage, the best chromosomes are choosen at random
        """
        for p in self.populations:
            for c in p.chromosomes:
                c.fitness = -np.inf

    def __getitem__(self, i):
        """
        Overload the index operation to return the i th population
        """
        return self.populations[i]

    def list_values(self):
        """
        Returns a list with elements being numpy arrays ecah containing the values of the genes
        of the best chromosome of each population obtained in the previous iteration
        """
        return [x.values() for x in self.best_chromosomes ]

    def array_values(self):
        """
        Returns an one dimensional numpy arrays comprised of the gene values obtained from
        the best chromosome of each population obtained in the previous iteration
        """
        return np.concatenate( self.list_values() )

    def population_no_genes(self):
        """
        Returns a list of the number of genes per chromosome for each population
        """
        return [len(x[0].variable_type) for x in self.populations]

    def list_values_replace(self, i, l):
        """
        Replace the list of gene values of the best chromosome of population i with l
        in the variable list obtained by list_values(). This is used in the fitness
        function evaluations.
        """
        v = self.list_values()
        r = v[0:i] + [np.array(l)] + v[i+1:]
        return r

    def array_values_replace(self, i, a):
        """
        Replace the array elements corresponding to the gene values of the
        best chromosome of population i with a in the numpy array returned by array_values().
        This is used in the fitness function evaluations.
        """
        no_genes = self.population_no_genes()
        lower = sum( no_genes[0:i] )
        upper = sum( no_genes[i+1:] )
        n = sum( no_genes )

        v = self.array_values()
        r = np.concatenate( [ v[0:lower], np.array(a), v[n - upper:] ] )
        return r

    def calc_on_arrays(self, c, i):
        """
        Calculate the fitness functions of chromosome c assuming it belongs to population i
        """
        c_values = c.values()
        v = self.array_values_replace(i, c_values)
        c.fitness = self.fitness_functions(i, v)
        return c.fitness

    def calc_arrayed_fitnesses(self):
        """
        Calculate the fitnesses assuming the fitness function are of type (ii), i.e.
        they operate directly on one dimensional numpy array containing the gene values
        of the best chromosomes
        """
        for i, p in enumerate(self.populations):
            for c in p.chromosomes:
                v = self.array_values_replace(i, c.values() )
                c.fitness = self.fitness_functions(i, v)
            p.sort_population()

    def calc_listed_fitnesses(self):
        """
        Calculate the fitnesses assuming the fitness function are of type (i), i.e.
        they operate on list of numpy arrays each containing the gene values
        of the best chromosomes of the populations
        """
        for i, p in enumerate(self.populations):
            for c in p.chromosomes:
                v = self.list_values_replace(i, c.values() )
                c.fitness = self.fitness_functions(v)[i]
            p.sort_population()

    def calc_fitnesses(self):
        """
        Calculate the fitnesses of all chromosomes in the group of populations
        """
        if self.calc_on_list:
            self.calc_listed_fitnesses()
        else:
            self.calc_arrayed_fitnesses()

    def next_generation(self):
        """
        Advances all populations by one generation and calculates
        the best chromosomes for each population to be used
        in the next turn.
        """
        for i, p in enumerate(self.populations):

            # We need to try to do this self.replacements_per_generation times

            for t in range(p.no_tournaments):


                # Tournament selection
                [p1, p2] = p.tournament_selection()

                # Cross over
                offspring = p1 + p2

                # Mutation
                offspring.mutate()

                # Fitness calculation
                self.calc_on_arrays(offspring, i)

                # Update pool
                is_replaced = p.replace( offspring )

                # If replacement has taken place adjust the number of replacements
                if is_replaced:
                    p.no_replacements += 1

                p.no_cross_overs += 1

            p.generations += 1

        self.calc_fitnesses()
        self.choose_best_chromosomes()
        self.generations += 1

    def __str__(self):
        """
        String equivalent of the class
        """
        msg = ''

        for p in self.populations:
            msg += str(p) + '\n'

        msg += 'Previously selected chromosomes:'
        for s in self.best_chromosomes:
            msg += str(s) + ', '

        return msg

    def __repr__(self):
        """
        Representation of the class
        """
        return self.__str__()

    def rel_fitness_ranges(self):
        """
        Calculate fitness range distribution for the various populations
        """
        return np.array([ c.rel_fitness_range() for c in self.populations ])

    def update_history(self):
        """
        Update history to keep the best chromosomes for each population
        """
        self.history_best.append( self.best_chromosomes )

    def fitness_evolution(self):
        """
        Return a 2D numpy array containing the fitness evolution of the various genes
        """
        n = self.generations
        m = self.no_populations
        f = np.zeros([n , m])
        for i, h in enumerate(self.history_best):
            f[i, :] = np.array([x.fitness for x in h])

        return f

    def terminate(self):
        """
        Check if the simulation needs to terminate
        """
        condition = (self.generations > self.max_generations) or \
                    ( all(self.rel_fitness_ranges() < self.min_rel_fitness_range ) )


        if hasattr(self.fitness_functions, 'calls'):
            condition = condition or (self.fitness_functions.calls > self.max_fitness_evals)

        return condition

    def report_best_chromosomes(self, verb_level):
        """
        print-out the best chromosomes of each population
        """
        self.report('Best chromosomes obtained so far:', verb_level, verb_level)

        for c in self.best_chromosomes:
            self.report(c, verb_level, include_stamp = False)

    def report_conditions(self, verb_level):
        """
        print-out the state of the termination conditions
        """
        ranges = self.rel_fitness_ranges()

        self.report('\nNumber of generations: %d' %self.generations, verb_level, include_stamp = False)

        if hasattr(self.fitness_functions, 'calls'):
            self.report('Number of fitness evaluations: %d' %self.fitness_functions.calls, verb_level, include_stamp = False)

        self.report('Relative fitness ranges: %s' %ranges, VERBOSITY_MEDIUM, include_stamp = False)

    def report_groups(self):
        """
        Report the state of the population groups
        """

        # Show best chromosomes
        self.report_best_chromosomes(VERBOSITY_LOW)

        # Show termination condition state
        self.report_conditions(VERBOSITY_MEDIUM)

        # Fitness range and individual state of each population in the group
        ranges = self.rel_fitness_ranges()
        self.report('\n', VERBOSITY_HIGH, include_stamp = False)
        for i, p in enumerate(self.populations):
            self.report('Population: %d' %i, VERBOSITY_HIGH, include_stamp = False)
            self.report(p, VERBOSITY_HIGH, include_stamp = False)
            self.report('Relative fitness range: %6.4f\n' %ranges[i], VERBOSITY_HIGH, include_stamp = False)

    def simulate(self):
        """
        Run the GA simulation until a termination condition is reached
        """
        while not(self.terminate() ):
            self.report('Now at generation: %d\n' %self.generations, VERBOSITY_MEDIUM)
            self.next_generation()
            self.report_groups()
            self.update_history()
            self.report('---------------------------------------------------------', VERBOSITY_MEDIUM, include_stamp = False)
