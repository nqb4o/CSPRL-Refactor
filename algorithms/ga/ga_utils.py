import torch
import torch.nn as nn
import numpy as np

class GAPolicy(nn.Module):
    """
    Simple MLP Policy for Genetic Algorithm / Neuroevolution.
    Matches the architecture used in the DQN implementation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(GAPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.net(x)

    def select_action(self, obs):
        with torch.no_grad():
            logits = self.forward(obs)
            return torch.argmax(logits).item()

def flatten_weights(model):
    """Convert model weights to a 1D chromosome."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def unflatten_weights(model, chromosome):
    """Load a chromosome back into the model."""
    start = 0
    for p in model.parameters():
        shape = p.data.shape
        size = np.prod(shape)
        p.data = torch.from_numpy(chromosome[start:start + size].reshape(shape)).float()
        start += size

def crossover(parent1, parent2, rate=0.5):
    """Uniform crossover."""
    mask = np.random.rand(*parent1.shape) < rate
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

def mutate(chromosome, rate=0.01, sigma=0.1):
    """Gaussian mutation."""
    mutation = np.random.randn(*chromosome.shape) * sigma
    mask = np.random.rand(*chromosome.shape) < rate
    chromosome[mask] += mutation[mask]
    return chromosome
