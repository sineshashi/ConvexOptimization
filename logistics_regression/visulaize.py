import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the data from CSV
data = pd.read_csv('input.csv')  # Replace 'data.csv' with your actual file name

# Extract features and target
X = data["x1"].values
y = data['target'].values

# Assume weights and bias are given
w = 4.04029096
b = -11.084617902330821

# Define the sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-w*x-b))

# Plot the data points
plt.scatter(X, y, color='blue', label='Class 0', s=20, alpha=0.7)

x1_vals = np.linspace(X.min(), X.max(), 100)
x2_vals = sigmoid(x1_vals)
plt.plot(x1_vals, x2_vals, color='green', label='Probability Function')

# Labels and legend
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.title('Logistic Regression Probability')
plt.show()
