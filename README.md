# Genetic Algorithm for Knapsack Problem

## Overview
This genetic algorithm is designed to solve the knapsack problem, which is a combinatorial optimization problem. The goal is to maximize the value of items placed in a knapsack without exceeding its weight capacity.

## Encoding of Individuals
Each individual (solution) is represented by a binary string, where each bit corresponds to an item. A bit value of `1` means the item is included in the knapsack, while a `0` means it's not.

## Genetic Operators
- **Crossover**: Uniform crossover is used, where each gene from the parent can be swapped with a fixed probability. This mixes the genetic information from two parents to create a child.
- **Mutation**: Bit flip mutation is applied, where each gene in the individual has a chance to be flipped from `0` to `1` or from `1` to `0`. This creates makes the population vary.

## Selection Method
Tournament selection is utilized, where groups of individuals compete against each other, and the best from each group is selected to breed and form the next generation.

## Fitness Function
The fitness of an individual is calculated as the total value of items in the knapsack. If the total weight exceeds the knapsack's capacity, the fitness is set to `0` to penalize the solution.

## Algorithm
The algorithm is run in parallel using a grid search to find the best combination of crossover probability, mutation probability, and number of generations. The fitness of each solution is tracked over generations to observe the evolution of the algorithm.

## How to run the algorithm
1. Clone the repository and navigate to the `Evo` directory.
2. Create a virtual environment using `python3 -m venv env`, activate it `source env/bin/activate`. This will help to keep the dependencies isolated.
3. Install the required packages using `pip install -r requirements.txt`.
4. Run the algorithm using `python3 main.py --file "Path to file with expected inpiut"` and analyze the output 
5. Deactive the virtual environment using `deactivate` and (optionally) remove it using `rm -rf env`.

## Results
Refer to report.txt. For us, "Fitness" is the total value of items in the knapsack.