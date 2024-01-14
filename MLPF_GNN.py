import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np

# Define the MLPF_GNN class
class MLPF_GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPF_GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # First Graph Convolution
        x = torch.relu(self.conv1(x, edge_index))
        
        # Second Graph Convolution
        x = torch.relu(self.conv2(x, edge_index))
        
        # Output layer
        x = self.conv3(x, edge_index)
        
        # Global Mean Pooling
        x = global_mean_pool(x, batch)  # Pooling for graph-level predictions
        return x

# Define the dimensions
input_dim = 3  # Example input features: hit position, energy deposit, etc.
hidden_dim = 64  # Hidden layer size
output_dim = 5  # Example output features: particle type, energy, etc.
num_graphs = 10  # Number of graphs (events) in a batch

# Create random data to simulate a batch of graphs (events)
edge_index = torch.randint(0, 100, (2, 1000), dtype=torch.long)  # Simulated connections between nodes
x = torch.rand((100, input_dim), dtype=torch.float)  # Simulated node features
batch = torch.randint(0, num_graphs, (100,), dtype=torch.long)  # Batch indices for each node

# Create the model and forward pass
model = MLPF_GNN(input_dim, hidden_dim, output_dim)
data = Data(x=x, edge_index=edge_index, batch=batch)

# Get the output from the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(data.x, data.edge_index, data.batch)

# Plotting
# Input features plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Input Features")
for i in range(input_dim):
    plt.hist(x[:, i].numpy(), bins=20, alpha=0.5, label=f'Feature {i+1}')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()

# Output features plot
plt.subplot(1, 2, 2)
plt.title("Output Features")
for i in range(output_dim):
    plt.hist(output[:, i].numpy(), bins=20, alpha=0.5, label=f'Feature {i+1}')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
