import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lstsqr_v1(x_ary, y_ary):
    """Computes the least-squares solution to a linear matrix equation."""
    cdef double x_avg, y_avg, var_x, cov_xy, slope, y_interc
    cdef double[:] x = x_ary # memory view
    cdef double[:] y = y_ary
    cdef long N

    N = x.shape[0]
    x_avg = np.sum(x)/N
    y_avg = np.sum(y)/N
    var_x = 0
    cov_xy = 0
    for i in range(N):
        temp = (x[i] - x_avg)
        var_x += temp**2
        cov_xy += temp*(y[i] - y_avg)
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)


# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double cysum(double[:] x):
    cdef:
        unsigned int i
        double s
        int N

    N = x.shape[0]

    for i in xrange(N):
        s += x[i]

    return s / N


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef lstsqr_v2(x_ary, y_ary):
    """Computes the least-squares solution to a linear matrix equation."""
    cdef double x_avg, y_avg, var_x, cov_xy,\
         slope, y_interc, temp, residuals, y_hat
    cdef double[:] x = x_ary # memory view
    cdef double[:] y = y_ary
    cdef long N
    cdef int i

    N = x.shape[0]
    x_avg = cysum(x) #np.sum(x)/N
    y_avg = cysum(y) #np.sum(y)/N
    var_x = 0
    cov_xy = 0
    residuals = 0
    for i in range(N):
        temp = (x[i] - x_avg)
        var_x += temp**2
        cov_xy += temp*(y[i] - y_avg)
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg

    for i in range(N):
        y_hat = x[i]*slope + y_interc
        residuals += (y[i] - y_hat)**2

    return (slope, y_interc, residuals)


# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef lstsqr_v3(x_ary, y_ary):
    """Computes the least-squares solution to a linear matrix equation.

    Assumes removal of the mean from the data and the design.
    """
    cdef double x_avg, y_avg, var_x, cov_xy, slope, residuals, y_hat
    cdef double[:] x = x_ary # memory view
    cdef double[:] y = y_ary
    cdef long N
    cdef int i

    N = x.shape[0]
    var_x = 0
    cov_xy = 0
    residuals = 0
    for i in range(N):
        var_x += x[i]**2
        cov_xy += x[i] * y[i]
    slope = cov_xy / var_x

    for i in range(N):
        y_hat = x[i]*slope
        residuals += (y[i] - y_hat)**2

    return (slope, residuals)


# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef lstsqr_v4(x_ary, y_ary):
    """Computes the least-squares solution to a linear matrix equation.

    Assumes removal of the mean from the data and the design.
    Does not compute residuals.
    """
    cdef double x_avg, y_avg, var_x, cov_xy, slope
    cdef double[:] x = x_ary # memory view
    cdef double[:] y = y_ary
    cdef long N
    cdef int i

    N = x.shape[0]
    var_x = 0
    cov_xy = 0
    for i in range(N):
        var_x += x[i]**2
        cov_xy += x[i] * y[i]
    slope = cov_xy / var_x

    return slope
