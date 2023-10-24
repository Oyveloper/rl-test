import numpy as np
import random
import torch
import gymnasium as gym
import wandb

N = 10
env = gym.vector.make("MountainCarContinuous-v0", num_envs=N)

total_num_episodes = int(6e3)
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

seed = 2


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
