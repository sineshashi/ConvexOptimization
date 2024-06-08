import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VALUES = [
    (3.08534041, 1.8834844491540403, 0),
    (2.46408285, 1.838742699110984, 50),
    (2.05108147, 1.8089991629876536, 100)
] #put the tuples (w1, b, lambda)

# Read data from CSV
df = pd.read_csv("input.csv")

# Plot positive data points
plt.scatter(df["x1"], df["target"], color="black", label="Points")

# Plot negative data points

# Generate x values
for w1, b, lmbda in VALUES:
    x_values = np.linspace(min(df["x1"]), max(df["x1"]), 10)

    # Calculate y values for the separating hyperplane
    y_values = (w1 * x_values + b)

    # Plot the separating hyperplane
    plt.plot(x_values, y_values, label=f'Lambda = {lmbda}')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
