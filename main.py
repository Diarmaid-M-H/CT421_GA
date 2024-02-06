import math
import random
import matplotlib.pyplot as plt


def generate_initial_solution(length):
    # generates a list containing numbers 0 to length -1.
    # For example: generate_initial_solution(5) returns [0] [1] [2] [3] [4]
    if length <= 0:
        raise Exception("Length must be greater than 0")

    return list(range(length))

def evaluate_fitness(solution):
    itemlist = problem.items
    """
    Evaluate the fitness of a solution based on the given rules.

    Parameters:
    - solution (list): A list of integers representing a solution.
    - problem (Problem): a problem instance containing relevant info like itemlist and bincapacity

    Returns:
    - int: The fitness score 
    """
    # Step 1: Initialize a dictionary to store the sum of values for each distinct element in the solution.
    distinct_values_sum = {}

    # Step 2: Iterate over each index and value in the solution.
    for index, value in enumerate(solution):
        # Step 3: Check if the value is already in the dictionary.
        if value not in distinct_values_sum:
            # If not, add the value as a key with the corresponding itemlist value as the initial sum.
            distinct_values_sum[value] = itemlist[index]
        else:
            # If the value is already in the dictionary, update the sum by adding the current itemlist value.
            distinct_values_sum[value] += itemlist[index]

    # Step 4: Initialize a variable to count the number of distinct values in the solution.
    distinct_count = len(set(solution))

    # Step 5: Iterate over the distinct values and check the sum of corresponding itemlist values.
    for value in distinct_values_sum:
        # If the sum is greater than bin capacity, return low fitness
        if distinct_values_sum[value] > problem.bin_capacity:
            return -1

    # Returns the max number of bins minus how many bins where used in this solution
    # This means fitness gets greater the less bins there are
    return len(itemlist) - distinct_count

def mutate(solution, mutation_rate):
    # change solution into a list
    # iterate through list, randomly swap 1 to 0 or 0 to 1
    mutated_solution = solution
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            # this can assign to any of the bins first generated
            # this may reduce in an empty bin being refilled (which could be optimal)
            # this can also overfill a bin which is combed out by having bad fitness
            mutated_solution[i] = random.randint(0, len(solution) - 1)
    return mutated_solution


def crossover(parent1, parent2):
    # chooses random int in the parent\
    # attaches "front" half of parent one to "back" half of parent 2, vice-versa
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def create_initial_population(population_size, solution_length):
    # creates <population_size> random solutions of length <solution_length>
    return [generate_initial_solution(solution_length) for _ in range(population_size)]


def calculate_average_fitness(population):
    # averages the fitness of all solutions
    total_fitness = sum(evaluate_fitness(solution) for solution in population)
    return total_fitness / len(population)

def elitism(population, elitism_percentage):
    population_size = len(population)
    elite_size = int(population_size * elitism_percentage)
    # sort population by fitness, fittest at start of list
    population.sort(key=evaluate_fitness, reverse=True)
    elite = population[:elite_size]
    # return the fittest individuals.
    return elite


def genetic_algorithm(mutation_rate, generations, elite_percentage, population_size):
    
    population = create_initial_population(population_size, len(problem.items))
    
    avg_fitness_history = []

    for generation in range(generations):
        # sort the solutions by fitness, with the most fit solutions at the start of the list
        population.sort(key=evaluate_fitness, reverse=True)

        avg_fitness = calculate_average_fitness(population)
        avg_fitness_history.append(avg_fitness)

        new_population = []
        elite_generation_size = int(math.ceil(population_size * elite_percentage))
        elite = population[:elite_generation_size]
        new_population.extend(elite)

        # TODO: everything below this change to be less exploitative
        for _ in range((population_size - len(elite) // 2)):
            parent1, parent2 = random.choices(population[:5], k=2)  # Select 2 parents from top 5 individuals
            # create 2 children by crossover from the 2 parents
            child1, child2 = crossover(parent1, parent2)
            # mutate the children
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            # add children to the new population
            new_population.extend([child1, child2])

        population = new_population

    return avg_fitness_history

class Problem:
    def __init__(self, name, num_item_weights, bin_capacity, items):
        self.name = name
        self.num_item_weights = num_item_weights
        self.bin_capacity = bin_capacity
        self.items = items


def parse_file(file_path):
    problems = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_problem = None
    for line in lines:
        line = line.strip()
        if line.startswith('\'BPP'):
            if current_problem:
                problems.append(current_problem)
            current_problem = Problem(name=line, num_item_weights=None, bin_capacity=None, items=[])
        elif current_problem and current_problem.num_item_weights is None:
            current_problem.num_item_weights = int(line)
        elif current_problem and current_problem.bin_capacity is None:
            current_problem.bin_capacity = int(line)
        elif current_problem:
            parts = line.split()
            item = int(parts[0])
            count = int(parts[1])
            current_problem.items.extend([item] * count)

    if current_problem:
        problems.append(current_problem)

    return problems

if __name__ == "__main__":
    population_size = 500
    mutation_rate = 0.01
    generations = 100
    elite_percentage = 0.01

    file_path = 'Binpacking.txt'
    problems = parse_file(file_path)
    #for problem in problems:

    global problem
    problem = problems[0]

    avg_fitness_history = genetic_algorithm(mutation_rate, generations, elite_percentage, population_size)

    # Plotting
    plt.plot(avg_fitness_history)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Genetic Algorithm: Bin Packing Problem")
    plt.show()
