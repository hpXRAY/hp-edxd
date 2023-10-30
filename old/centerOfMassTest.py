import numpy as np
import matplotlib.pyplot as plt
import time

def find_peak_center(data, num_points=6):
    n = len(data)
    x = np.arange(n)

    # Calculate the background using an average of multiple points at the start and end
    background_start = np.mean(data[:num_points])
    background_end = np.mean(data[-num_points:])
    # Create a trimmed data array to match the background start and end positions
    trimmed_data = data[num_points//2:-num_points//2]
    # Create a corresponding x array for the trimmed data
    trimmed_x = x[num_points//2:-num_points//2]
    # Calculate a linear background for the trimmed data
    background = (background_end - background_start) / (len(trimmed_data) - 1) * (trimmed_x - trimmed_x[0]) + background_start
    data_adjusted = trimmed_data - background
    # Normalize the data to 1
    normalized_data = data_adjusted / np.max(data_adjusted)
    # Find the FWHM points
    half_max = 0.5
    above_half = normalized_data > half_max
    fwhm_points = np.where(above_half)[0]
    fwhm_center = int((fwhm_points[0] + fwhm_points[-1]) // 2.0)
    # Create a tighter background within FWHM distance from the new peak center
    fwhm_distance = int((fwhm_points[-1] - fwhm_points[0])*1.5 )
    background_start_index = fwhm_center - fwhm_distance 
    background_end_index = fwhm_center + fwhm_distance 
    background_start = normalized_data[background_start_index]
    background_end = normalized_data[background_end_index]
    fwhm_surrounding_indexes = fwhm_center - fwhm_distance + np.arange(2* fwhm_distance+1)
    x_tight = trimmed_x[fwhm_surrounding_indexes]
    tighter_background = (background_end - background_start) / (2 * fwhm_distance) * (x_tight-x_tight[0] ) + background_start
    normalized_data_tight = normalized_data[fwhm_surrounding_indexes]
    data_adjusted_tight = normalized_data_tight - tighter_background
    data_squared = data_adjusted_tight**2
    # Compute the center of mass of the square of the data, squaring suppresses the contribution from background
    center_of_mass = np.sum(x_tight * data_squared) / np.sum(data_squared)
    return center_of_mass , fwhm_points, x_tight, data_adjusted_tight

# Sample data with a Gaussian peak and linear background
n = 100
x = np.arange(n)
background = 0.05 * x
peak = np.exp(-(x - 40) ** 2 / (2 * 10 ** 2))
data = background + peak + 0.01 * np.random.randn(n)

# Find the center of mass, background, trimmed_x, and data with background subtracted
now = time.time()
fwhm_center, fwhm_points, x_tight, data_adjusted_tight = find_peak_center(data, num_points=6)
later = time.time()
print('elapsed ',later-now)

print('Center of Mass = ', fwhm_center)


# Create subplots in one window
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first subplot: Data with Background and Center of Mass
axes[0].plot(x, data, label='Simulated Data')
axes[0].axvline(x=fwhm_center, color='red', linestyle='--', label='Center of Mass (Original)')
axes[0].set_xlabel('Position')
axes[0].set_ylabel('Intensity')
axes[0].set_title('Data with Background and Center of Mass')
axes[0].legend()

# Plot the second subplot: Data with Background Subtracted
axes[1].plot(x_tight, data_adjusted_tight, label='Data with Background Subtracted', color='green')
#axes[1].axvline(x=fwhm_center, color='red', linestyle='--', label='Center of Mass (Original)')
axes[1].axvline(x=fwhm_center, color='purple', linestyle='--', label='Center of Mass (Tightened Background)')
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Intensity')
axes[1].set_title('Data with Background Subtracted and Center of Mass')
axes[1].legend()

# Plot the second subplot: Data with Background Subtracted
# Adjust layout and display
plt.tight_layout()
plt.show()