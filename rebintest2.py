import numpy as np
import matplotlib.pyplot as plt
 
# Unevenly spaced x-y data with duplicate x values
x = np.array([0.0, 0.1, 0.5, 0.3, .9, 1.0, 1.1, 1.9, 2.0, 2.1, 2.5, 2.6, 3.0, 3.0, 3.5, 3.5, 4.0, 4.1, 4.5, 4.5])
noise = np.random.normal(0, 0.5, len(x)) *0.5
y = np.sin(x) + noise
weights = np.random.uniform(0.5, 1.5, len(x))


# Determine the number of bins
num_bins = 10

# Calculate the bin edges
bin_edges = np.linspace(np.min(x), np.max(x), num_bins + 1)

# Initialize empty lists for rebinned data
rebin_x = []
rebin_y = []

# Distribute data into bins
for i in range(num_bins):
    mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
    if np.sum(weights[mask]) != 0:  # Check if the weights sum to zero
        rebin_x.append((bin_edges[i] + bin_edges[i + 1]) / 2)  # Use bin centers as rebinned x values
        rebin_y.append(np.average(y[mask], weights=weights[mask]))  # Weighted average of y values within each bin


# Compute the step size between values in rebin_x
step = np.min(np.diff(rebin_x))

# Construct a new evenly spaced rebin_x array
new_rebin_x = np.arange(rebin_x[0], rebin_x[-1], step)

# Interpolate rebin_y values using numpy's interp function
new_rebin_y = np.interp(new_rebin_x, rebin_x, rebin_y)

# Plot the original data with colors based on weights
plt.scatter(x, y, c=weights, cmap='coolwarm', label='Original Data')

# Plot the rebinned data as connected lines with round markers
plt.plot(new_rebin_x, new_rebin_y, 'r.-', label='Rebinned Data')

# Draw vertical lines to indicate bin boundaries
for bin_edge in bin_edges:
    plt.axvline(x=bin_edge, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data and Rebinned Data with Interpolation')
plt.legend()
plt.colorbar(label='Weight')
plt.show()