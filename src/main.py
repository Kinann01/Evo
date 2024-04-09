import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import product

def get_file():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str, help="Path to file with expected input", required=True)
    args = parser.parse_args()
    return args.file

def parse_input(input_file_name):
    print("Parsing input file: {}".format(input_file_name))
    with open(input_file_name, 'r') as file:
        num_items, max_weight = map(int, file.readline().strip().split())
        items = np.array([line.strip().split() for line in file], dtype=int)
    print("--------")
    print("Parsing complete.")     
    return max_weight, num_items, items
        
def eval_knapsack(individual, items, max_weight):
    individual_np = np.array(individual)
    weight = sum(individual_np * items[:, 1])
    value = sum(individual_np * items[:, 0])
    if weight > max_weight:
        return 0,
    return value,

def run_evolutionary_algorithm(pop_size, cxpb, mutpb, num_generations, max_weight, num_items, items):

    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("zeroOne", lambda : 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.zeroOne, n=num_items)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_knapsack, items=items, max_weight=max_weight)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    _, logs = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=num_generations, stats=stats, halloffame=hof, verbose=False)
    return cxpb, mutpb, num_generations, hof[0].fitness.values[0], hof, logs

def grid_search(max_weight, num_items, items):

    parameter_grid = { # Grid used for smaller tests
        'cxpb': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'mutpb': [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.8],
        'ngen': [50, 100]
    }

    # parameter_grid = { # Grid used for 1000 items
    #     'cxpb': [0.4, 0.7, 0.8, 0.9],
    #     'mutpb': [0.01, 0.1, 0.5, 0.8],
    #     'ngen': [50, 100]
    # }

    best_fitness = float('-inf')
    best_params = None
    pop_size = num_items * 15
    best_i = None

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_evolutionary_algorithm, pop_size, cxpb, mutpb, ngen, max_weight, num_items, items)
                   for cxpb, mutpb, ngen in product(*parameter_grid.values())]

        for future in futures:
            cxpb, mutpb, ngen, fitness, hof, _ = future.result()
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = (cxpb, mutpb, ngen)
                best_i = hof

    print("Grid search complete.")
    print("Individual {}".format(best_i))
    print("Best Parameters:\ncxpb={}\nmutpb={}\nngen={}\nFitness={}\n".format(best_params[0], best_params[1], best_params[2], best_fitness))
    return best_params

def run_manually_for_plotting(max_weight, num_items, items, cxpb, mutpb, ngen):
    _, _, _, _, _, logs = run_evolutionary_algorithm(num_items*15, cxpb, mutpb, ngen, max_weight, num_items, items)
    
    max_fitness = logs.select("max")
    avg_fitness = logs.select("avg")    
    plt.plot(max_fitness, label="Maximum Fitness")
    plt.plot(avg_fitness, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file = get_file()
    max_weight, num_items, items = parse_input(file)
    best_params = grid_search(max_weight, num_items, items)
    run_manually_for_plotting(max_weight, num_items, items, *best_params)

if __name__ == "__main__":
    main()