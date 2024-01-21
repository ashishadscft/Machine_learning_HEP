import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class TTRegressionModel(nn.Module):
    def __init__(self, input_size, regression_outputs, classification_outputs):
        super(TTRegressionModel, self).__init__()
        self.regression_branch = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, regression_outputs)
        )
        
        self.classification_branch = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, classification_outputs),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        regression_output = self.regression_branch(x)
        classification_output = self.classification_branch(x)
        return regression_output, classification_output

# Create the model
input_size = 100  # Replace with actual size
regression_outputs = 4  # Px, Py, Pz, E for neutrino
classification_outputs = 3  # Signal HH, DY, TT
model = TTRegressionModel(input_size, regression_outputs, classification_outputs)

# Loss functions
regression_loss_fn = nn.MSELoss()
classification_loss_fn = nn.CrossEntropyLoss()

# Combine losses for joint optimization
def combined_loss(regression_pred, regression_true, classification_pred, classification_true):
    regression_loss = regression_loss_fn(regression_pred, regression_true)
    classification_loss = classification_loss_fn(classification_pred, classification_true)
    # Placeholder for mass loss - you will need to implement the actual computation
    mass_loss = torch.tensor(0.0)  # Replace with actual mass loss computation
    return regression_loss + classification_loss + mass_loss

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:  # Assuming you have a DataLoader for your data
        inputs, regression_targets, classification_targets = batch
        optimizer.zero_grad()
        regression_pred, classification_pred = model(inputs)
        loss = combined_loss(regression_pred, regression_targets, classification_pred, classification_targets)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
