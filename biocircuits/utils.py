import numpy as np


class AttributeContainer(object):
    """Generic class to hold attributes."""

    def __init__(self, **kw):
        self.__dict__ = kw


def nearly_zero_to_zero(x, tol=100.0):
    """
    Convert all elements in an array that are nearly zero to zero.

    Parameters
    ----------
    x : array_like
        Array with entires that are possibly close to zero
    tol : float, default 100.0
        If a number is within a factor of `tol` to machine precision,
        set to zero.

    Returns
    -------
    output : array_like
        The array `x` with entries close to zero set to exactly zero.
    """
    return (np.abs(x) > tol * np.finfo(float).eps) * x


def findiff(x, y, D=None, order=1, accuracy=2):
    """
    Computes a finite differencing stencil and stores it as a matrix
    given by sparse_type. Central differencing formulae are used
    everywhere except one-sided differencing is used to maintain the
    same order of accuracy at the boundaries.

    Parameters
    ----------
    xgrid : array_like
        The positions of the grid points for which the stencil is to be
        used.
    order : int
        The order of the derivative
    accuracy : int, default 2
        The order of the accuracy for the differencing stencil. I.e.,
        the error for computing the derivative at a point with
        len(xgrid) = N is order N^(-accuracy). Accuracy must be an even
        number.

    Returns
    -------
    output : 2D array
        Finite difference stencil D of sparse type sparse_type. If `y`
        is an array of values evaluated at `xgrid`, then the derivative
        of `y` is computed as `np.dot(output, y)`.
    """
    if D is None:
        D = fd_stencile(x, order, accuracy=accuracy)




def fd_stencil(xgrid, order, accuracy=2):
    """
    Computes a finite differencing stencil and stores it as a matrix
    given by sparse_type. Central differencing formulae are used
    everywhere except one-sided differencing is used to maintain the
    same order of accuracy at the boundaries.

    Parameters
    ----------
    xgrid : array_like
        The positions of the grid points for which the stencil is to be
        used.
    order : int
        The order of the derivative
    accuracy : int, default 2
        The order of the accuracy for the differencing stencil. I.e.,
        the error for computing the derivative at a point with
        len(xgrid) = N is order N^(-accuracy). Accuracy must be an even
        number.

    Returns
    -------
    output : 2D array
        Finite difference stencil D of sparse type sparse_type. If `y`
        is an array of values evaluated at `xgrid`, then the derivative
        of `y` is computed as `np.dot(output, y)`.
    """
    # Number of grid points
    n = len(xgrid)

    # Check input
    if accuracy % 2 != 0:
        raise RuntimeError("Must have accuracy be an even integer.")

    # Set maximum accuracy based on number of grid points
    if accuracy > n - 1:
        if n % 2 == 0:
            accuracy = n - 2
        else:
            accuracy = n - 1

    # Convenient to have around
    a2 = accuracy // 2

    # Number of grid points used in differencing calculation
    n_points = accuracy + 1

    # Initialize the matrix
    D = np.zeros((n, n))

    # Build the middle section
    for i in range(a2 + 1, n - a2 + 1):
        x = xgrid[i - a2 - 1 : i + a2]
        w = _fd_weights(x[a2], x, order)
        w = nearly_zero_to_zero(w)
        D[i - 1, i - a2 - 1 : i + a2] = w

    # Build end sections
    for i in range(1, a2 + 1):
        # Left section
        x = xgrid[0:n_points]
        w = _fd_weights(x[i - 1], x, order)
        w = nearly_zero_to_zero(w)
        D[i - 1, 0:n_points] = w

        # Right section
        x = xgrid[n - n_points : n]
        w = _fd_weights(x[n_points - i], x, order)
        w = nearly_zero_to_zero(w)
        D[n - i, n - n_points : n] = w

    return D


def _fd_weights(xi, x, m):
    """
    Compute the finite difference weights for computing the mth
    derivative at point `xi` using data given at points specified by
    `x`. E.g., if we had grid points at ..., -2, -1, 0, 1, 2, ... and we
    wanted to compute the FD weights for computing a derivative at 0
    using central differencing, we would use:
      xi = 0.0
      x = np.array([-1.0, 0.0, 1.0])
      m = 1

    Parameters
    ----------
    xi : float
        Evaluation point for the derivative (scalar)
    x : array_like
        Nodes of differencing grid
    m : int
        Order of derivative sought

    Returns
    -------
    output : 1D array
        Weights for the approximation to the mth derivative

    Notes
    -----
    .. This is a compact implementation, NOT an efficient one! See
    Fornberg's "A Practical Guide to Pseudospectral Methods", Cambridge
    Univ. Press, page 18.
    """

    n = len(x) - 1
    w = np.zeros(n + 1)
    x = x - xi  # Translate evaluation point to zero
    for k in range(0, n + 1):
        w[k] = _point_weight(x, m, n, k)

    return w


def _point_weight(x, m, j, k):
    """
    Recursion for the FD weights, assuming evaluation point is zero.

    Parameters
    ----------
    x : array_like
        Nodes for finite difference calculation
    m : int
        Order of derivative sought
    j : int
        Stencil width (i.e., use first j+1 nodes only)
    k : int
        Index of node for this weight (in 0:j)

    Returns
    -------
    output : float
        Finite difference weight.

    Notes
    -----
    .. For algorithmic description, see Fornberg, A Practical Guide to
    Pseudospectral Methods, page 18.
    """
    # Undefined coefficients
    if m < 0 or m > j:
        c = 0.0
    # Base case of one-point interpolation
    elif m == 0 and j == 0:
        c = 1.0
    # Generic recursion
    else:
        if k < j:
            c = (
                x[j] * _point_weight(x, m, j - 1, k)
                - m * _point_weight(x, m - 1, j - 1, k)
            ) / (x[j] - x[k])
        else:
            beta = np.prod(x[j - 1] - x[0 : j - 1]) / np.prod(x[j] - x[0:j])
            c = beta * (
                m * _point_weight(x, m - 1, j - 1, j - 1)
                - x[j - 1] * _point_weight(x, m, j - 1, j - 1)
            )

    return c


def _convert_data(data, inf_ok=False, min_len=1):
    """
    Convert inputted 1D data set into NumPy array of floats.
    All nan's are dropped.

    Parameters
    ----------
    data : int, float, or array_like
        Input data, to be converted.
    inf_ok : bool, default False
        If True, np.inf values are allowed in the arrays.
    min_len : int, default 1
        Minimum length of array.

    Returns
    -------
    output : ndarray
        `data` as a one-dimensional NumPy array, dtype float.
    """
    # If it's scalar, convert to array
    if np.isscalar(data):
        data = np.array([data], dtype=np.float)

    # Convert data to NumPy array
    data = np.array(data, dtype=np.float)

    # Make sure it is 1D
    if len(data.shape) != 1:
        raise RuntimeError("Input must be a 1D array or Pandas series.")

    # Remove NaNs
    data = data[~np.isnan(data)]

    # Check for infinite entries
    if not inf_ok and np.isinf(data).any():
        raise RuntimeError("All entries must be finite.")

    # Check to minimal length
    if len(data) < min_len:
        raise RuntimeError(
            "Array must have at least {0:d} non-NaN entries.".format(min_len)
        )

    return data


