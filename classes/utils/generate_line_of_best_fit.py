import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
df = pd.read_csv(input("Input the path to the CSV file to draw the line of best fit for: "))
x = df['k_values']
y = df['precision']

# Calculate the coefficients (slope and intercept) of the best-fit line
# The '1' indicates a first-degree polynomial (linear)
coefficients = np.polyfit(x, y, 1) 
print(coefficients)

print(df['precision'][9])
print(df['precision'][49])
print(df['precision'][69])

# # Create a polynomial function from the coefficients
# # This allows you to easily calculate y-values for the line
# poly_function = np.poly1d(coefficients)

# # Generate y-values for the best-fit line
# y_line = poly_function(x)

# # Plot the original data points
# plt.scatter(x, y, label='Data Points')

# # Plot the line of best fit
# plt.plot(x, y_line, color='red', label='Line of Best Fit')

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line of Best Fit')
# plt.legend()
# plt.grid(True)
# plt.show()