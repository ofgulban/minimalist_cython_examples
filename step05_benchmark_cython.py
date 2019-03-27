"""Least squares with cython."""

import numpy as np
import time
from step03_cython_lstsqr import lstsqr_v1, lstsqr_v2, lstsqr_v3, lstsqr_v4

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
# Cython least squares
start = time.time()
results = []
for x in range(nr_voxels):
    slope, _ = lstsqr_v1(design[:, 0], data[x, :])
    results.append(slope)

print("Cython function 1")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))

# -----------------------------------------------------------------------------
# Cython least squares 2
start = time.time()
results = []
for x in range(nr_voxels):
    slope, _, _ = lstsqr_v2(design[:, 0], data[x, :])
    results.append(slope)

print("Cython function 2")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))

# -----------------------------------------------------------------------------
# Demean data and design matrix
data_demean = data - np.mean(data, axis=1)[:, None]
design_demean = design[:, 0] - np.mean(design[:, 0])

# Cython least squares 3, remove mean
start = time.time()
results = []
for x in range(nr_voxels):
    slope, _ = lstsqr_v3(design_demean, data_demean[x, :])
    results.append(slope)

print("Cython function 3")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))

# -----------------------------------------------------------------------------
# Cython least squares 4, remove mean, dont compute residuals
start = time.time()
results = []
for x in range(nr_voxels):
    slope = lstsqr_v4(design_demean, data_demean[x, :])
    results.append(slope)

print("Cython function 4")
print("  {}".format(results[:3]))
print("  Duration: {:.2f} sec".format(time.time() - start))
