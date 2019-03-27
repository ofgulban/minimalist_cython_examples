"""Least squares in python."""

import numpy as np
import time

# Create data
nr_voxels, nr_timepoints = 10000, 400
data = np.random.random([nr_voxels, nr_timepoints])

# Create design matrix
design = np.ones((nr_timepoints, 2))
design[:, 0] = np.random.rand(nr_timepoints)

print("---")
print("Nr. voxels: {}".format(nr_voxels))
print("Nr. time points: {}".format(nr_timepoints))
print("---")


# =============================================================================
def python_lstsqr(x_list, y_list):
    """Compute the least-squares solution to a linear matrix equation."""
    N = len(x_list)
    x_avg = sum(x_list)/N
    y_avg = sum(y_list)/N
    var_x, cov_xy = 0, 0
    for x, y in zip(x_list, y_list):
        temp = x - x_avg
        var_x += temp**2
        cov_xy += temp * (y - y_avg)
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)


# Python least squares
start = time.time()
results = []
for x in range(nr_voxels):
    slope, interc = python_lstsqr(design[:, 0], data[x, :])
    results.append(slope)

print("Python function")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))
