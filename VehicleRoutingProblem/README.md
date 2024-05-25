
## Vehicle Routing Problem (VRP) Solver Using Ant Colony Optimization (ACO)

This project implements an Ant Colony Optimization (ACO) algorithm to solve the Vehicle Routing Problem (VRP). 

## Changes Made to the Original Code

The following changes were added:

1) XML Parsing: Added XML parsing to read input data (nodes, vehicles, and requests) from an XML file.

2) Fitness Function: Modified the fitness function to consider multiple routes and vehicle capacities.

3) Solution Generation: Adapted solution generation to construct multiple routes per vehicle.

4) Pheromone Update: Updated pheromone update rules to account for multiple routes.

5) Grid Search: Implemented a grid search to find the best parameters for the algorithm based on the size of the graph.

## Running the Program

### With Parameters

You can run the program with specific parameters using command-line arguments:

```bash
python3 main.py --file path_to_input.xml --ants 100 --max_iter 200 --alpha 1 --beta 2 --Q 50 --rho 0.6 --seed 42
```

- `--file`: Path to the XML input file containing nodes, vehicles, and requests. (Required)
- `--ants`: Number of ants.
- `--max_iter`: Number of iterations.
- `--alpha`: Alpha parameter (pheromone influence).
- `--beta`: Beta parameter (heuristic influence).
- `--Q`: Q parameter (pheromone deposit amount).
- `--rho`: Rho parameter (pheromone evaporation rate).
- `--seed`: Seed for random number generation (optional, for reproducibility).

### Without Parameters (Grid Search)

If you run the program without specifying the parameters, a grid search will be performed to find the best parameters:

```bash
python3 main.py --file path_to_input.xml
```

The grid search will automatically determine the optimal parameters based on the size of the graph (number of nodes).

## Report.txt

`report.txt` contains the results for all three input files. For `data_32` a grid search was used to find the best parameters. while for `data_72` `data_422`, the parameters were manually set using command-line arguments since the grid search was taking too long.

## graphs

`graphs` directory contains graphs for the three input files. The graphs show convergence of the algorithm for each input file over the number of iterations.