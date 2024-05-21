# NEAT Neuroevolution for Gym Environments

## Environments

### CartPole-v1
The goal of the CartPole environment is to balance a pole on a moving cart. The agent receives rewards for keeping the pole upright for as long as possible. The episode ends when the pole falls over or the cart moves too far from the center.

### MountainCar-v0
The MountainCar environment requires the agent to drive a car up a hill. The car is underpowered and cannot reach the goal directly, so the agent must learn to drive back and forth to build momentum and reach the goal. The agent receives a reward of -1 for each time step.

## How to Run

Install Dependencies

```sh
pip install gym neat-python numpy matplotlib
```

Run the Script

```sh
python script_name.py --env CartPole-v1 --generations 50 --episodeDuration 500 --numEpisodes 5 --configPath path/to/config-file
```

- `--env`: The game environment to use (e.g., "CartPole-v1" or "MountainCar-v0").
- `--generations`: Number of generations to run.
- `--episodeDuration`: Duration of each episode.
- `--numEpisodes`: Number of episodes to average fitness over.
- `--configPath`: (Optional) Path to the NEAT configuration file. If not provided, a default path based on the environment will be used.
