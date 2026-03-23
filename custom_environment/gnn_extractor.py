import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for SB3 to process graph observations.
    Expects observation_space to be a spaces.Dict with:
      - node_features: (N, F)
      - edge_index: (2, E)
      - global_state: (1,)
    """
    def __init__(self, observation_space, features_dim=256):
        super(GNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        n_node_features = observation_space.spaces["node_features"].shape[1]
        self.n_nodes = observation_space.spaces["node_features"].shape[0]
        
        # Simple two-layer GCN
        self.gcn1_weight = nn.Parameter(torch.Tensor(n_node_features, 64))
        self.gcn1_bias = nn.Parameter(torch.Tensor(64))
        
        self.gcn2_weight = nn.Parameter(torch.Tensor(64, 128))
        self.gcn2_bias = nn.Parameter(torch.Tensor(128))
        
        nn.init.xavier_uniform_(self.gcn1_weight)
        nn.init.zeros_(self.gcn1_bias)
        nn.init.xavier_uniform_(self.gcn2_weight)
        nn.init.zeros_(self.gcn2_bias)
        
        self.linear = nn.Linear(128 + 1, features_dim)

    def forward(self, observations):
        x = observations["node_features"]      # (B, N, F)
        edge_index = observations["edge_index"].long() # (B, 2, E)
        global_state = observations["global_state"]    # (B, 1)
        
        B = x.shape[0]
        device = x.device
        
        # We assume the graph structure is static across the batch elements.
        edges = edge_index[0] # (2, E)
        
        # Build dense adjacency matrix A
        adj = torch.zeros((self.n_nodes, self.n_nodes), device=device)
        adj[edges[0], edges[1]] = 1.0
        
        # Add self loops
        adj += torch.eye(self.n_nodes, device=device)
        
        # Normalize adj: D^-0.5 A D^-0.5
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        
        # Expand to batch (B, N, N)
        norm_adj = norm_adj.unsqueeze(0).expand(B, -1, -1)
        
        # Layer 1 message passing
        support1 = torch.matmul(x, self.gcn1_weight) # (B, N, 64)
        out1 = torch.bmm(norm_adj, support1) + self.gcn1_bias
        out1 = torch.relu(out1)
        
        # Layer 2 message passing
        support2 = torch.matmul(out1, self.gcn2_weight) # (B, N, 128)
        out2 = torch.bmm(norm_adj, support2) + self.gcn2_bias
        out2 = torch.relu(out2)
        
        # Global mean pooling over nodes
        graph_embed = out2.mean(dim=1) # (B, 128)
        
        # Combine with global budget
        # global_state shape is (B, 1) -> flatten usually required by SB3 extractors, but already (B, 1)
        combined = torch.cat([graph_embed, global_state], dim=1) # (B, 129)
        
        return torch.relu(self.linear(combined))
