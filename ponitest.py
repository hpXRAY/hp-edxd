import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(x, poni_x, poni_angle, distance):
    if np.array_equal(x, poni_x):
        return poni_angle
    elif np.all(x < poni_x):
        return poni_angle - np.arctan((poni_x - x) / distance)
    elif np.all(x > poni_x):
        return poni_angle + np.arctan((x - poni_x) / distance)
    else:
        # Handle the case where x contains a mix of values less and greater than poni_x
        result = np.empty_like(x, dtype=float)
        result[x == poni_x] = poni_angle
        result[x < poni_x] = poni_angle - np.arctan((poni_x - x[x < poni_x]) / distance)
        result[x > poni_x] = poni_angle + np.arctan((x[x > poni_x] - poni_x) / distance)
        return result

# Define the inputs
x = np.linspace(0, 191, 192)
poni_x = 96
poni_angle = 5 * np.pi / 180
distance = 4000

# Calculate the angles
result = calculate_angle(x, poni_x, poni_angle, distance)

# Plot the results
plt.plot(x, result)
plt.xlabel('x')
plt.ylabel('Angle (radians)')
plt.title('Angle vs. x')
plt.grid(True)
plt.show()
