import numpy as np
import scipy.integrate
import scipy.interpolate


def ddeint(func, y0, t, tau, args=(), y0_args=(), n_time_points_per_step=None):
    """Integrate a system of delay differential equations defined by
        y' = f(t, y, y(t-tau1), y(t-tau2), ...)
    using the method of steps. All tau's are assumed constant.

    Parameters
    ----------
    func : function, call signature func(y, t, y_past, *args)
        Function specifying the right hand side of the system of DDEs.
        Assuming we have a system of `n` DDEs, its arguments are:
        y : Numpy array, shape (n, )
            The current value of y.
        t : float
            The current time.
        y_past : function, call signature y_past(t)
            Function used to compute values of y in the past. This is
            not specified by the user, but called as, e.g.,
            `y_past(t-tau)` within `func`. The function is automatically
            generated using interpolants.
        args : tuple
            Tuple of any other arguments to be passed `func`. Note that
            the values of the delays, `tau`, are usually included in
            `args`.
    y0 : function, call signature y0(t, *y0_args)
        A function to compute the pre- time = t[0] values of `y`.
    t : array_like
        The time points for which the solution of the DDEs is to be
        returned.
    tau : float or array_like
        Set of time delays for DDEs. Only the shortest and longest are
        used.
    args : tuple, default ()
        Tuple of arguments to be passed to `func`.
    y0_args : tuple, default ()
        Tuple of arguments to be passed to `y0`.
    n_time_points_per_step : int, default 200
        The number of time points to store the solution for each step.
        These points are then used to compute an interpolant.

    Returns
    -------
    y : array, shape (len(t), len(y0))
    Array containing the value of y for each desired time in t.

    Notes
    -----
    .. Uses `scipy.integrate.odeint()` to integrate each step. To
       compute the values of `y` at time points from a previous step,
       uses a cubic B-spline interpolant of the solution from the
       previous step.
    .. `n_time_points_per_step` may be adjusted downward if the value
       of `y` does not change rapidly for a given step, but should be
       large enough to be able to capture the dynamics.

    """
    if np.isscalar(tau):
        tau = np.array([tau])
    else:
        tau = np.array(tau)

    if (tau <= 0).any():
        raise RuntimeError("All tau's must be greater than zero.")

    tau_short = np.min(tau)
    tau_long = np.max(tau)

    if n_time_points_per_step is None:
        n_time_points_per_step = max(
            int(1 + len(t) / (t.max() - t.min()) * tau_long), 20
        )

    t0 = t[0]

    # Past function for the first step
    y_past = lambda time_point: y0(time_point, *y0_args)

    # Integrate first step
    t_step = np.linspace(t0, t0 + tau_short, n_time_points_per_step)
    y = scipy.integrate.odeint(func, y_past(t0), t_step, args=(y_past,) + args)

    # Store result from integration
    y_dense = y.copy()
    t_dense = t_step.copy()

    # Get dimension of problem for convenience
    n = y.shape[1]

    # Integrate subsequent steps
    j = 1
    while t_step[-1] < t[-1]:
        t_start = max(t0, t_step[-1] - tau_long)
        i = np.searchsorted(t_dense, t_start, side="left")
        t_interp = t_dense[i:]
        y_interp = y_dense[i:, :]

        # Make B-spline
        tck = [scipy.interpolate.splrep(t_interp, y_interp[:, i]) for i in range(n)]

        # Interpolant of y from previous step
        y_past = (
            lambda time_point: np.array(
                [scipy.interpolate.splev(time_point, tck[i]) for i in range(n)]
            )
            if time_point > t0
            else y0(time_point, *y0_args)
        )

        # Integrate this step
        t_step = np.linspace(
            t0 + j * tau_short, t0 + (j + 1) * tau_short, n_time_points_per_step
        )
        y = scipy.integrate.odeint(func, y[-1, :], t_step, args=(y_past,) + args)

        # Store the result
        y_dense = np.append(y_dense, y[1:, :], axis=0)
        t_dense = np.append(t_dense, t_step[1:])

        j += 1

    # Interpolate solution for returning
    y_return = np.empty((len(t), n))
    for i in range(n):
        tck = scipy.interpolate.splrep(t_dense, y_dense[:, i])
        y_return[:, i] = scipy.interpolate.splev(t, tck)

    return y_return
