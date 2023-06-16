'''To perform weighted rebinning of unevenly spaced data into evenly spaced bins, you can follow these steps:

Sort the input arrays based on the x-values.
Calculate the cumulative sum of the weights.
Determine the bin edges for the evenly spaced bins.
Compute the indices of the input data points that fall into each bin.
Calculate the weighted sum of the y-values for each bin.
Compute the weighted average y-values for each bin.
Create evenly spaced x-values for the bins.
Plot the rebinned data.'''

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import numpy as np
from scipy.interpolate import interp1d

# Unevenly spaced x-y data with duplicate x values
x = np.array      ([  0,   1, 1.1,    3,   5,   7,   8,   9,  11,  12,12.1,12.2,  13,  14,  16])  # Unevenly spaced x-values with duplicates
y = np.array      ([  1,   2,   3,    5,   6,   8,   5,   4,   3,   2,   9,   9,   5,   6,   7])  # Corresponding y-values
weights = np.array([0.5, 1.5, 0.5,  0.8, 1.2, 0.7, 0.5, 1.5, 0.5,   1,   1,   1, 0.7, 1.2, 0.7])  # Associated weights

'''x = np.array([0, 1, 1, 3, 5, 7])
y = np.array([1, 2, 3, 4, 6, 8])
weights = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.7])'''

# Sort the arrays based on x values
sort_indices = np.argsort(x)
sorted_x = x[sort_indices]
sorted_y = y[sort_indices]
sorted_weights = weights[sort_indices]

# Sort the arrays based on x values
sort_indices = np.argsort(x)
sorted_x = x[sort_indices]
sorted_y = y[sort_indices]
sorted_weights = weights[sort_indices]

# Determine the bin edges for evenly spaced bins
bin_edges = np.linspace(sorted_x[0], sorted_x[-1], num=6)

# Accumulate the weighted sum and cumulative weights for each bin
bin_weighted_sums = np.zeros_like(bin_edges[1:])
cumulative_weights = np.zeros_like(bin_edges[1:])
bin_indices = np.digitize(sorted_x, bin_edges, right=True)
for i, bin_index in enumerate(bin_indices[:-1]):
    bin_weighted_sums[bin_index] += sorted_y[i] * sorted_weights[i]
    cumulative_weights[bin_index] += sorted_weights[i]

# Compute the weighted average y-values for each bin
valid_indices = cumulative_weights > 0
bin_weighted_averages = bin_weighted_sums / np.maximum(cumulative_weights, 1e-15)

# Create evenly spaced x-values for the bins
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p1 = win.addPlot( title="interpolated")
#p1.plot(unique_x,unique_y, pen=None, symbol='t', symbolBrush=(0,255,0, 80), symbolPen=None)
p1.plot(x,y, pen=None, symbolBrush=(255,0,0), symbolPen=None)
p1.plot(bin_centers,bin_weighted_averages, pen=(100,100,100), symbolBrush=(0,0,255), symbolPen=None, symbolSize=5)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
