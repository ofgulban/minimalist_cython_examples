"""Least squares with numpy."""

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
# Numpy least squares
start = time.time()
results = []
for x in range(nr_voxels):
    slope, interc, = np.linalg.lstsq(design, data[x, :].T, rcond=None)[0]
    results.append(slope)

print("Numpy function")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))

# -----------------------------------------------------------------------------
# Vectorized numpy least squares
start = time.time()
slope, _ = np.linalg.lstsq(design, data.T, rcond=None)[0]
results = list(slope)

print("Numpy with vectorization")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))
