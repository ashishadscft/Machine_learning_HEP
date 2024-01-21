import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Regression model
class TTRegressionModel(nn.Module):
    def __init__(self):
        super(TTRegressionModel, self).__init__()
        # Define the layers for the regression part
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        # Output layer for neutrino momenta
        self.out_reg = nn.Linear(128, 6)  # Assuming 6 momenta components

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        momenta = self.out_reg(x)
        return momenta

# Define the Classification model with skip connections
class TTClassificationModel(nn.Module):
    def __init__(self, regression_model):
        super(TTClassificationModel, self).__init__()
        self.regression_model = regression_model
        # Define the layers for the classification part
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.skip = nn.Linear(128, 128)  # Skip connection
        # Output layer for classification
        self.out_class = nn.Linear(128, 3)  # Assuming 3 classes

    def forward(self, x):
        reg_out = self.regression_model(x)
        x = F.relu(self.fc1(reg_out))
        x = F.relu(self.fc2(x))
        # Implement skip connections
        x = F.relu(self.fc3(x)) + self.skip(reg_out)
        x = F.relu(self.fc4(x)) + self.skip(reg_out)
        classes = self.out_class(x)
        return classes, reg_out

# Combine both models
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.regression_model = TTRegressionModel()
        self.classification_model = TTClassificationModel(self.regression_model)

    def forward(self, x):
        classes, reg_out = self.classification_model(x)
        momenta = self.regression_model(reg_out)
        return classes, momenta

# Example of using the Combined Model
in_features = 4  # Number of input features (replace with actual number)
combined_model = CombinedModel()

# Assuming we have an example input tensor
input_tensor = torch.randn(10, in_features)  # Replace 10 with the actual batch size

# Forward pass through the combined model
class_pred, momenta_pred = combined_model(input_tensor)

# Define the loss function
criterion = nn.MSELoss()  # Replace with actual loss function

# Example loss calculation (replace targets with actual data)
regression_targets = torch.randn(10, 6)  # Replace 10 with the actual batch size
classification_targets = torch.empty(10, dtype=torch.long).random_(3)  # Replace 10 with the actual batch size

# Calculate losses separately
regression_loss = criterion(momenta_pred, regression_targets)
classification_loss = F.cross_entropy(class_pred, classification_targets)

# Combine losses
total_loss = regression_loss + classification_loss  # Add other components as needed

# Example backward pass
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
