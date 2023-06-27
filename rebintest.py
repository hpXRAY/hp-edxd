import numpy as np
import matplotlib.pyplot as plt

def quadratic_sequence(start, end, num_points):
    x = np.linspace(0, 1, num_points)
    y = x**2
    sequence = start + (end - start) * y
    sequence[-1] = end
    return sequence

start_value = 0.5
end_value = 2.5
num_points = 6

result = quadratic_sequence(start_value, end_value, num_points)

# Generate x-coordinates for plotting
x = np.linspace(0, 1, num_points)

# Plot the sequence
plt.plot(x, result, 'o-')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Quadratic Sequence')
plt.grid(True)
plt.show()