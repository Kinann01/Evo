# NEAT Neuroevolution for Gym Environments

## Environments

### CartPole-v1
The goal of the CartPole environment is to balance a pole on a moving cart. The agent receives rewards for keeping the pole upright for as long as possible. The episode ends when the pole falls over, the cart moves too far from the center, or the maximum number of time steps is reached.

### MountainCar-v0
The MountainCar environment requires the agent to drive a car up a hill. The car is underpowered and cannot reach the goal directly, so the agent must learn to drive back and forth to build momentum and reach the goal. The agent receives a reward of -1 for each time step.

## How to Run

Install Dependencies

```sh
pip install gym neat-python numpy matplotlib
```

Run the Script

```sh
cd src
python3 main.py --env CartPole-v1 --generations 50 --episodeDuration 500 --numEpisodes 5 --configPath path/to/config-file
```

- `--env`: The game environment to use (e.g., "CartPole-v1" or "MountainCar-v0").
- `--generations`: Number of generations to run.
- `--episodeDuration`: Duration of each episode.
- `--numEpisodes`: Number of episodes to average fitness over.
- `--configPath`: (Optional) Path to the NEAT configuration file. If not provided, a default path based on the environment will be used.


## Fitness Evaluation 

The fitness function evaluates each genome (neural network) by running it in the specified environment for a number of episodes. The fitness is calculated based on the rewards accumulated during these episodes.


CartPole-v1: The fitness is the average reward over multiple episodes. The reward is the number of time steps the pole remains balanced.

MountainCar-v0: The fitness is the average reward over multiple episodes. The reward is -1 for each step taken.

## Results

The script outputs the best genome after the specified number of generations and shows its performance in the environment. The fitness over generations is plotted to visualize the evolution process. Navigate to the `results` directory to view the plots. The plots are plotted for default arguments of `--generations 50 --episodeDuration 500 --numEpisodes 5`.