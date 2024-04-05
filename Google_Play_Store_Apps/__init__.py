# Importing the matplotlib library for plotting
import matplotlib.pyplot as plt

# Define the fractions (x-axis values) and accuracy scores (y-axis values) for each type of data corruption
fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gaussian_noise_accuracies = [0.78, 0.80, 0.84, 0.82, 0.73, 0.68, 0.65, 0.60, 0.55, 0.52, 0.50]
missing_values_accuracies = [0.78, 0.82, 0.83, 0.80, 0.79, 0.78, 0.61, 0.68, 0.54, 0.51, 0.48]
scaling_accuracies = [0.78, 0.78, 0.79, 0.79, 0.78, 0.73, 0.72, 0.67, 0.62, 0.53, 0.49]

# Plotting the accuracies for Gaussian noise
plt.plot(fractions, gaussian_noise_accuracies, marker='o', linestyle='-', color='b', label='Gaussian Noise')

# Plotting the accuracies for missing values
plt.plot(fractions, missing_values_accuracies, marker='o', linestyle='-', color='r', label='Missing Values')

# Plotting the accuracies for scaling errors
plt.plot(fractions, scaling_accuracies, marker='o', linestyle='-', color='g', label='Scaling')

# Adding title and labels for the axes
plt.title('Model Accuracy vs. Fraction of Data Corruption')
plt.xlabel('Fraction of Data Corruption')
plt.ylabel('Model Accuracy')

# Adding a legend to explain which line corresponds to which type of data corruption
plt.legend()

# Displaying the grid for better readability
plt.grid(True)

# Showing the plot
plt.show()
