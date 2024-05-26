from collections import namedtuple
import math
import functools
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
import argparse
import matplotlib.pyplot as plt
import random

# static_counter = 0

Node = namedtuple('Node', ['id', 'type', 'x', 'y'])
Vehicle = namedtuple('Vehicle', ['departure', 'arrival', 'capacity'])
Request = namedtuple('Request', ['id', 'node', 'quantity'])

def parse_input(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    nodes = {}
    for node in root.findall('.//node'):
        id = int(node.get('id'))
        type = int(node.get('type'))
        x = float(node.find('cx').text)
        y = float(node.find('cy').text)
        nodes[id] = Node(id, type, x, y)

    xml_vehicle = root.find('.//vehicle_profile')
    departure = int(xml_vehicle.find('departure_node').text)
    arrival = int(xml_vehicle.find('arrival_node').text)
    capacity = float(xml_vehicle.find('capacity').text)
    vehicle = Vehicle(departure, arrival, capacity)

    requests = []
    for request in root.findall('.//request'):
        id = int(request.get('id'))
        node = int(request.get('node'))
        quantity = float(request.find('quantity').text)
        requests.append(Request(id, node, quantity))

    return nodes, vehicle, requests

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# main ACO function for VRP
def ant_solver_vrp(nodes, vehicle, requests, distance, ants=10, max_iter=3000, alpha=1, beta=3, Q=100, rho=0.8, seed=None):
    if seed is not None:
        set_seed(seed)
    
    node_indices = {node_id: index for index, node_id in enumerate(nodes.keys())}
    P = initialize_pheromone(len(nodes))
    best_sol = None
    best_fit = float('inf')
    fitness_history = []
    
    for it in range(max_iter):
        sols = list(generate_solutions(nodes, vehicle, requests, P, distance, node_indices, ants, alpha=alpha, beta=beta))
        fits = list(map(lambda x: fitness(nodes, distance, x), sols))
        P = update_pheromone(P, sols, fits, Q=Q, rho=rho, node_indices=node_indices)        
        for s, f in zip(sols, fits):
            if f < best_fit:
                best_fit = f
                best_sol = s
        
        fitness_history.append(best_fit)
        print('Iteration {}: Best Fitness = {}'.format(it, best_fit))
    
    return best_sol, P, fitness_history


# compute distance
@functools.lru_cache(maxsize=None)
def distance(n1, n2):
    return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

# compute fitness
def fitness(nodes, dist, sol):
    total_distance = 0
    for route in sol:
        route_distance = 0
        for i in range(len(route) - 1):
            route_distance += dist(nodes[route[i]], nodes[route[i + 1]])
        route_distance += dist(nodes[route[-1]], nodes[route[0]])  # return to depot
        total_distance += route_distance
    return total_distance

# pheromone initialization
def initialize_pheromone(N):
    return 0.01 * np.ones(shape=(N, N))

# generate solutions
def generate_solutions(nodes, vehicle, requests, P, dist, node_indices, N, alpha=1, beta=3):
    def compute_prob(n1, n2):
        nu = 1 / dist(nodes[n1], nodes[n2])
        tau = P[node_indices[n1], node_indices[n2]]
        ret = pow(tau, alpha) * pow(nu, beta)
        return ret if ret > 0.000001 else 0.000001

    depot = vehicle.departure

    for _ in range(N):
        available = [req.node for req in requests]
        routes = []
        while available:
            route = [depot]
            current_load = 0
            while available and current_load < vehicle.capacity:
                probs = np.array([compute_prob(route[-1], node_id) for node_id in available])
                selected = np.random.choice(available, p=probs / sum(probs))
                request_quantity = next(req.quantity for req in requests if req.node == selected)
                if current_load + request_quantity <= vehicle.capacity:
                    route.append(selected)
                    current_load += request_quantity
                    available.remove(selected)
                else:
                    break

            route.append(depot)
            routes.append(route)

        yield routes

def update_pheromone(P, sols, fits, Q=100, rho=0.6, node_indices=None):
    ph_update = np.zeros(shape=P.shape)
    for s, f in zip(sols, fits):
        for route in s:
            for i in range(len(route) - 1):
                ph_update[node_indices[route[i]]][node_indices[route[i + 1]]] += Q / f
            ph_update[node_indices[route[-1]]][node_indices[route[0]]] += Q / f
    
    return (1 - rho) * P + ph_update

def gridSearch(nodes, vehicle, requests, distance, numNodes):
    parameter_grid = {}

    if numNodes <= 50:
        parameter_grid = {
            'ants': [5, 10, 20],
            'max_iter': [100, 200],
            'alpha': [1, 2, 3],
            'beta': [2, 3],
            'Q': [20, 50, 75],
            'rho': [0.3, 0.7]
        }
    elif numNodes > 50 and numNodes <= 200:
        parameter_grid = {
            'ants': [20, 50, 150],
            'max_iter': [200, 500],
            'alpha': [1, 3],
            'beta': [2, 4],
            'Q': [50, 150],
            'rho': [0.3, 0.7]
        }
    else:
        parameter_grid = {
            'ants': [200, 300, 400],
            'max_iter': [500, 700, 1000],
            'alpha': [1, 2, 3, 4],
            'beta': [2, 3, 4, 5],
            'Q': [100, 150, 200],
            'rho': [0.2, 0.4, 0.6]
        }

    best_fit = float('inf')
    best_params = None
    results = []

    with ProcessPoolExecutor() as executor:
        futures = []
        param_list = []
        for ants in parameter_grid['ants']:
            for max_iter in parameter_grid['max_iter']:
                for alpha in parameter_grid['alpha']:
                    for beta in parameter_grid['beta']:
                        for Q in parameter_grid['Q']:
                            for rho in parameter_grid['rho']:
                                params = (ants, max_iter, alpha, beta, Q, rho)
                                future = executor.submit(ant_solver_vrp, nodes, vehicle, requests, distance, *params, seed=42)  # Ensure consistent seed
                                futures.append(future)
                                param_list.append(params)

        for future, params in zip(futures, param_list):
            try:
                sol, _, _ = future.result()
                fit = fitness(nodes, distance, sol)
                results.append((fit, params))
                if fit < best_fit:
                    best_fit = fit
                    best_params = params
                print(f"Tested parameters {params} with fitness {fit}")
            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")

    return best_params



def plot_convergence(fitness_history):
    plt.figure()
    plt.plot(fitness_history, label='Best Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('ACO Convergence')
    plt.legend()
    plt.show()

def main():
    
    args = None
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem') 
    parser.add_argument('--file', type=str, help='Path to XML file', required=True)
    parser.add_argument('--ants', type=int, help='Number of ants', default=None)
    parser.add_argument('--max_iter', type=int, help='Number of iterations', default=None)
    parser.add_argument('--alpha', type=int, help='Alpha parameter', default=None)
    parser.add_argument('--beta', type=int, help='Beta parameter', default=None)
    parser.add_argument('--Q', type=int, help='Q parameter', default=None)
    parser.add_argument('--rho', type=float, help='Rho parameter', default=None)
    args = parser.parse_args()

    nodes, vehicle, requests = parse_input(args.file)
    numNodes = len(nodes)
    best_sol = None
    best_params = None
    f = None

    if args.ants is not None: ## All the parameters have to be filled
        best_sol, _, fitnessHistory = ant_solver_vrp(nodes, vehicle, requests, distance, ants=args.ants, max_iter=args.max_iter, alpha=args.alpha, beta=args.beta, Q=args.Q, rho=args.rho)
        f = fitnessHistory
    else:
        best_params = gridSearch(nodes, vehicle, requests, distance, numNodes)
        print("Best parameters Found: {}".format(best_params)) 
    
    if (best_sol is None):
        best_sol, _, fitnessHistory = ant_solver_vrp(nodes, vehicle, requests, distance, *best_params, seed=None)
        f = fitnessHistory

    print("Best solution: {}".format(best_sol))
    print("Fitness: {}".format(fitness(nodes, distance, best_sol)))
    plot_convergence(f)

if __name__ == '__main__':
    main()