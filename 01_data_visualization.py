"""This version uses PyTorch for a simple neural network mockup to replace the TensorFlow section."""

# ==============================
# AI / ML COMMON IMPORTS
# ==============================

# Numerical computing
import numpy as np

# Data handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Machine Learning (classical) - You need to run: pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Deep Learning (PyTorch replacement for TensorFlow)
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# SAMPLE DEMOGRAPHIC DATA
# ==============================

ages = [22, 25, 47, 52, 46, 56, 48, 30, 34, 29, 41, 38, 27, 33, 36]
income = [30000, 42000, 65000, 72000, 69000, 80000, 73000, 48000, 52000, 46000, 61000, 58000, 45000, 51000, 56000]
genders = ["Male", "Female", "Female", "Male", "Male", "Female", "Male",
           "Female", "Male", "Female", "Male", "Female", "Female", "Male", "Female"]

# Convert to DataFrame
data = pd.DataFrame({
    "Age": ages,
    "Income": income,
    "Gender": genders
})

# ==============================
# 1️⃣ HISTOGRAM — Age Distribution
# ==============================
plt.figure()
plt.hist(data["Age"], bins=5, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ==============================
# 2️⃣ PIE CHART — Gender Distribution
# ==============================
gender_counts = data["Gender"].value_counts()

plt.figure()
plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", colors=['lightcoral', 'lightblue'])
plt.title("Gender Distribution")
plt.show()

# ==============================
# 3️⃣ BAR GRAPH — Average Income by Gender
# ==============================
avg_income = data.groupby("Gender")["Income"].mean()

plt.figure()
plt.bar(avg_income.index, avg_income.values, color=['lightcoral', 'lightblue'])
plt.title("Average Income by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Income")
plt.show()

# ==============================
# 4️⃣ SCATTER PLOT — Age vs Income
# ==============================
plt.figure()
plt.scatter(data["Age"], data["Income"], color='purple')
plt.title("Age vs Income")
plt.xlabel("Age")
plt.ylabel("Income")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==============================
# 5️⃣ PYTORCH DEMO (Simple Linear Regression)
# ==============================
# Convert data to PyTorch tensors
X = torch.tensor(data["Age"].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data["Income"].values, dtype=torch.float32).view(-1, 1)

# Define a simple Linear Model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input (Age), One output (Income)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict using the trained model
predicted = model(X).detach().numpy()

# Visualization of the PyTorch Model Fit
plt.figure()
plt.scatter(data["Age"], data["Income"], color='purple', label='Actual Data')
plt.plot(data["Age"], predicted, color='red', label='PyTorch Prediction')
plt.title("Income Prediction (PyTorch Linear Regression)")
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()