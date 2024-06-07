import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data from CSV
df = pd.read_csv("input.csv")

# Plot positive data points
plt.scatter(df[df["output"]==1]["x1"], df[df["output"]==1]["x2"], color="blue", label="Positive +1")

# Plot negative data points
plt.scatter(df[df["output"]==-1]["x1"], df[df["output"]==-1]["x2"], color="red", label="Negative -1")

# Define weights and bias for the separating hyperplane

w1 = float(input("Weight vector first component: "))
w2 = float(input("Weight vector second component: "))
b = float(input("Bias term: "))
# w1, w2 = 13.33333331, -6.66666665
# b = -21.6666666436286

# Generate x values
x_values = np.linspace(min(df["x1"]), max(df["x1"]), 10)

# Calculate y values for the separating hyperplane
y_values = (-w1 * x_values - b) / w2

# Plot the separating hyperplane
plt.plot(x_values, y_values, linestyle='--', color='green', label='Separating Hyperplane')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
