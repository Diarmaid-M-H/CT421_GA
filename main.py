import math
import random
import matplotlib.pyplot as plt


def generate_random_solution(length):
    # generates a string of length (length) randomly filled with 1s and 0s
    return ''.join(random.choice('01') for _ in range(length))


def evaluate_fitness(solution):
    # fitness is measured by the distance between the target string and the solution (small value = more fit)
    if len(solution) != len(target_solution):
        raise ValueError("Strings being compared must be of equal length")

    # fitness is higher at a lower hamming distance (when the solutions are more similar)
    return len(solution) - (sum(bit1 != bit2 for bit1, bit2 in zip(solution, target_solution)))


def mutate(solution, mutation_rate):
    # change solution into a list
    # iterate through list, randomly swap 1 to 0 or 0 to 1
    mutated_solution = list(solution)
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = '1' if solution[i] == '0' else '0'
    return ''.join(mutated_solution)


def crossover(parent1, parent2):
    # chooses random int in the parent\
    # attaches "front" half of parent one to "back" half of parent 2, vice-versa
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def create_initial_population(population_size, solution_length):
    # creates <population_size> random solutions of length <solution_length>
    return [generate_random_solution(solution_length) for _ in range(population_size)]


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


def genetic_algorithm(population_size, solution_length, mutation_rate, generations, elite_percentage, target_solution):
    population = create_initial_population(population_size, solution_length)
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


if __name__ == "__main__":
    population_size = 500
    solution_length = 100
    mutation_rate = 0.01
    generations = 20
    elite_percentage = 0.01
    target_solution = "1000010010010000111100111100111001110100110011101100001110110110100001101111010110010110101011110011"

    avg_fitness_history = genetic_algorithm(population_size, solution_length, mutation_rate, generations,
                                            elite_percentage, target_solution)

    # Plotting
    plt.plot(avg_fitness_history)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Genetic Algorithm: One-Max Problem")
    plt.show()
