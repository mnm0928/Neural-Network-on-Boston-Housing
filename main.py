import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'california_housing.csv')

# Split data into features and target
X = df[df.columns[1:-1]].values #This slicing operation selects a subset of columns starting from the second column (index 1) up to, but not including, the last column.
y = df['MedHouseVal'].values

print("shape of X:", X.shape)
print("shape of y:", y.shape)

# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(8, 16)
        self.activation1 = activation_fn()
        self.layer2 = nn.Linear(16, 32)
        self.activation2 = activation_fn()
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.layer3(x)
        return x

# Step 7: Train the model for ReLU activation and find the optimal learning rate
def train_model_relu(model, criterion, optimizer, X_train, y_train, epochs, lr):
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

# Train the model for TanH activation and find the optimal learning rate
def train_model_tanh(model, criterion, optimizer, X_train, y_train, epochs, lr):
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

model_relu = SimpleNN(activation_fn=nn.ReLU)
model_tanh = SimpleNN(activation_fn=nn.Tanh)
criterion = nn.MSELoss()
optimizer = optim.SGD
learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
mse_list_relu = []
mse_list_tanh = []

# Train the ReLU model and find the optimal learning rate
for lr in learning_rates:
    model_relu = SimpleNN(activation_fn=nn.ReLU)
    train_model_relu(model_relu, criterion, optimizer, X_train, y_train, epochs=20, lr=lr)

    # Evaluate the model on the test set
    model_relu.eval()
    y_pred_relu = model_relu(X_test)
    mse_relu = criterion(y_pred_relu, y_test.view(-1, 1))
    mse_list_relu.append(mse_relu)
    print(f"Learning Rate (ReLU): {lr}, Mean Squared Error: {mse_relu}")

# Train the TanH model and find the optimal learning rate
for lr in learning_rates:
    model_tanh = SimpleNN(activation_fn=nn.Tanh)
    train_model_tanh(model_tanh, criterion, optimizer, X_train, y_train, epochs=20, lr=lr)

    # Evaluate the model on the test set
    model_tanh.eval()
    y_pred_tanh = model_tanh(X_test)
    mse_tanh = criterion(y_pred_tanh, y_test)
    mse_list_tanh.append(mse_tanh)
    print(f"Learning Rate (Tanh): {lr}, Mean Squared Error: {mse_tanh}")

print(f"\nComparison")
print(f"ReLU: Avg of MSE: {sum(mse_list_relu)/8.0}")
print(f"Tanh: Avg of MSE: {sum(mse_list_tanh)/8.0}")



