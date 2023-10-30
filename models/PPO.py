from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import os

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x) -> torch.tensor:
        return self.backbone(x)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.mean_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, output_dim), std=1.0),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(output_dim)))


    def forward(self, x) -> Tuple[torch.tensor, torch.tensor]:
        action_mean = self.mean_network(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd


class PPO(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.critic = CriticNetwork(input_dim)


    def get_value(self, state: torch.tensor) -> torch.tensor:
        return self.critic(state)

    def sample_action(self, observation: np.ndarray, action = None) -> np.ndarray:
        action_mean, action_logstd = self.policy(observation)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(observation)


    def save(self, base_location: str):
        if not os.path.exists(base_location):
            os.makedirs(base_location)
        torch.save(self.policy, f"{base_location}/policy.pt")
        torch.save(self.critic, f"{base_location}/critic.pt")

    def load(self, base_location: str):
        self.policy = torch.load(f"{base_location}/policy.pt")
        self.critic = torch.load(f"{base_location}/critic.pt")