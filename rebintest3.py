import numpy as np
import matplotlib.pyplot as plt

# Unevenly spaced x-y data with duplicate x values
x = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.0, 4.5, 4.5])
noise = np.random.normal(0, 0.5, len(x))
y = np.sin(x) + noise
weights = np.random.uniform(0.5, 1.5, len(x))

# Randomize the order of the points
random_indices = np.random.permutation(len(x))
x = np.sort(x[random_indices])
y = y[random_indices]
weights = weights[random_indices]

# Step 1: Sort the input arrays based on the x-values
sort_indices = np.argsort(x)
sorted_x = x[sort_indices]
sorted_y = y[sort_indices]
sorted_weights = weights[sort_indices]

# Step 2: Determine the bin edges for the evenly spaced bins
num_bins = len(np.unique(sorted_x))  # Adjust num_bins to match the number of unique x-values
bin_edges = np.linspace(np.min(sorted_x) - 1e-6, np.max(sorted_x), num=num_bins + 1)  # Adjusted bin edges

# Step 3: Calculate the bin indices for each data point
bin_indices = np.digitize(sorted_x, bin_edges, right=True) - 1

# Step 4: Compute the weighted sum and count for each bin
bin_weighted_sums = np.bincount(bin_indices, weights=sorted_y * sorted_weights, minlength=num_bins)
bin_counts = np.bincount(bin_indices, minlength=num_bins)

# Step 5: Compute the weighted average y-values for each bin
bin_weighted_averages = np.divide(bin_weighted_sums, bin_counts, out=np.zeros_like(bin_weighted_sums),
                                   where=bin_counts != 0)

# Step 6: Create evenly spaced x-values for the bins
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

# Identify the missing bins
missing_bins = np.where(bin_counts == 0)[0]

# Interpolate the missing bins using linear interpolation
for missing_bin in missing_bins:
    # Find the nearest non-missing bins
    left_bin = np.max(np.where(bin_counts[:missing_bin] != 0)[0])
    right_bin = np.min(np.where(bin_counts[missing_bin:] != 0)[0]) + missing_bin

    # Interpolate the y-values
    left_y = bin_weighted_averages[left_bin]
    right_y = bin_weighted_averages[right_bin]
    interpolated_y = np.interp(bin_centers[missing_bin], [bin_centers[left_bin], bin_centers[right_bin]], [left_y, right_y])
    bin_weighted_averages[missing_bin] = interpolated_y

# Plot the original data with colors based on weights
plt.scatter(x, y, c=weights, cmap='coolwarm', label='Original Data')

# Plot the rebinned data
plt.plot(bin_centers, bin_weighted_averages, 'r.-', label='Rebinned Data')

# Draw vertical lines to indicate bin boundaries
for bin_edge in bin_edges:
    plt.axvline(x=bin_edge, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data and Weighted Rebinning with Interpolation')
plt.legend()
plt.colorbar(label='Weight')
plt.show()
