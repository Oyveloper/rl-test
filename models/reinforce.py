from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

        self.head = nn.Sequential(nn.Linear(8, output_size), nn.Softmax(1))

        # self.mean_network = nn.Sequential(
        #     nn.Linear(16, output_size),
        #     nn.ReLU(),
        # )

        # self.variance_network = nn.Sequential(nn.Linear(16, output_size), nn.ReLU())

    def forward(self, x: torch.tensor) -> torch.tensor:
        base = self.backbone(x.float())

        # mean = self.mean_network(base)
        # variance = torch.log(1 + torch.exp(self.variance_network(base)))

        return self.head(base)


class REINFORCE:
    def __init__(self, observation_size: int, action_size: int):
        self.policy_network = PolicyNetwork(observation_size, action_size)

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.optimizer = torch.optim.AdamW(
            self.policy_network.parameters(), lr=self.learning_rate
        )

    def sample_action(self, observation: np.ndarray) -> int:
        state = torch.tensor(np.array([observation]))
        probs = self.policy_network(state)
        # mean, variance = self.policy_network(state)

        # distribution = torch.distributions.normal.Normal(
        #     mean[0] + self.eps, variance[0] + self.eps
        # )
        action = probs.multinomial(num_samples=1).item()
        # action = distribution.sample()
        # prob = distribution.log_prob(action)

        self.probs.append(torch.log(probs[0][action]))

        return action

    def save(self, location: str):
        torch.save(self.policy_network, location)

    def load(self, location: str):
        self.policy_network = torch.load(location)

    def update(self):
        running_g = 0
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)
        deltas = torch.tensor(gs)

        loss = 0

        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []
