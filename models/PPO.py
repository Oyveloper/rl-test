import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, output_dim * 2),
            nn.Tanh(),
        )

    def forward(self, x) -> torch.tensor:
        return self.backbone(x)


class PPO:
    def __init__(self, input_dim: int, output_dim: int):
        self.policy = PolicyNetwork(input_dim, output_dim)
         
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6
        
        self.probs = []
        self.rewards = []
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.learning_rate
        )
        
    def sample_action(self, observation: np.ndarray) -> np.ndarray:
        state = torch.tensor(np.array([observation]))

        pred = self.policy(state)
        mean = pred[0]
       
        std = pred[1]
        distribution = torch.distributions.normal.Normal(
            mean + self.eps, std + self.eps
        )
        action = distribution.sample
        prob = distribution.log_prob(action)
        
        self.probs.append(prob)
        
        return action
    
    def save(self, location: str):
        torch.save(self.policy, location)
        
    def load(self, location: str):
        self.policy = torch.load(location)
    
    def update(self):
        