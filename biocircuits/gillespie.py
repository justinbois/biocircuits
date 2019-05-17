import multiprocessing
import warnings

import numpy as np
import numba

try:
    import tqdm

    has_tqdm = True
except:
    has_tqdm = False


@numba.njit
def _sample_discrete(probs, probs_sum):
    q = np.random.rand() * probs_sum

    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1


@numba.njit
def _sum(ar):
    return ar.sum()


@numba.njit
def _draw_time(props_sum):
    return np.random.exponential(1 / props_sum)


def _gillespie_draw(propensity_func, propensities, population, t, args):
    """
    Draws a reaction and the time it took to do that reaction.
    """
    # Update propensities
    propensity_func(propensities, population, t, *args)

    # Sum of propensities
    props_sum = _sum(propensities)

    # Compute time
    time = _draw_time(props_sum)

    # Draw reaction given propensities
    rxn = _sample_discrete(propensities, props_sum)

    return rxn, time


def _gillespie_trajectory(
    propensity_func, update, population_0, time_points, draw_fun, args=()
):
    # Number of species
    n_species = update.shape[1]

    @numba.njit
    def _copy_population(population_previous, population):
        for i in range(n_species):
            population_previous[i] = population[i]

    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int64)

    # Initialize and perform simulation
    j_time = 1
    j = 0
    t = time_points[0]
    population = population_0.copy()
    population_previous = population_0.copy()
    pop_out[0, :] = population
    propensities = np.zeros(update.shape[0])
    while j < len(time_points):
        while t < time_points[j_time]:
            # draw the event and time step
            event, dt = draw_fun(propensity_func, propensities, population, t, args)

            # Update the population
            _copy_population(population_previous, population)
            population += update[event, :]

            # Increment time
            t += dt

        # Update the index (Have to be careful about types for Numba)
        j = np.searchsorted((time_points > t).astype(np.int64), 1)

        # Update the population
        for k in np.arange(j_time, min(j, len(time_points))):
            pop_out[k, :] = population_previous

        # Increment index
        j_time = j

    return pop_out, None


@numba.njit
def _gillespie_trajectory_report_time_points(
    propensity_func, update, population_0, time_points, draw_fun, args=()
):
    # Number of iterations before concatenating arrays
    n_iter = 1000

    # Initialize arrays for storing trajectories
    pop = np.empty((n_iter, update.shape[1]), dtype=np.int64)
    tp = np.empty(n_iter, dtype=np.float64)

    # Initialize and perform simulation
    t = time_points[0]
    population = population_0.copy()
    time_points_out = np.array([t])
    pop_out = population_0.copy().reshape((1, len(population)))
    propensities = np.zeros(update.shape[0])
    j = 0
    while t < time_points[-1]:
        i = 0
        while i < n_iter and t < time_points[-1]:
            # draw the event and time step
            event, dt = draw_fun(propensity_func, propensities, population, t, args)

            # Update the population
            population += update[event, :]
            pop[i, :] = population

            # Increment time
            t += dt
            tp[i] = t

            # Increment indexes
            i += 1
            j += 1

        # Add this subtrajectory to output
        pop_out = np.concatenate((pop_out, pop))
        time_points_out = np.concatenate((time_points_out, tp))

        # Reset index
        i = 0

    return pop_out[:j, :], time_points_out[:j]


def _gillespie_ssa(
    propensity_func,
    update,
    population_0,
    time_points,
    return_time_points=False,
    size=1,
    args=(),
    progress_bar=False,
):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.

    Parameters
    ----------
    propensity_func : function
        Function with call signature 
        `propensity_func(propensities, population, t, *args) that takes
        the current propensities and population of particle counts and 
        updates the propensities for each reaction. It does not return 
        anything.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
    return_time_points : bool, default False
        If True, returns a trajectory and the time points of the 
        trajectory, going from time_points[0] to time_points[-1].
    size : int, default 1
        Number of trajectories to sample.
    args : tuple, default ()
        The set of parameters to be passed to propensity_func.
    progress_bar : str or bool, default False
        If True, use standard tqdm progress bar. If 'notebook', use
        tqdm.notebook progress bar. If False, no progress bar.

    Returns
    -------
    sample : ndarray, shape (size, num_time_points, num_chemical_species)
        Entry i, j, k is the count of chemical species k at time
        time_points[j] for trajectory i.
    """
    # Get number of species
    n_species = update.shape[1]

    # Make sure input population has correct dimensions
    if n_species != len(population_0):
        raise RuntimeError(
            "Number of rows in `update` must equal length of `population_0."
        )

    @numba.njit
    def _copy_population(population_previous, population):
        for i in range(n_species):
            population_previous[i] = population[i]

    # Build trajectory function based on if propensity function is jitted
    if type(propensity_func) == numba.targets.registry.CPUDispatcher:

        @numba.njit
        def _draw(propensities, population, t):
            """
            Draws a reaction and the time it took to do that reaction.
            """
            # Compute propensities
            propensity_func(propensities, population, t, *args)

            # Sum of propensities
            props_sum = np.sum(propensities)

            # Compute time
            time = np.random.exponential(1 / props_sum)

            # Draw reaction given propensities
            rxn = _sample_discrete(propensities, props_sum)

            return rxn, time

        if return_time_points:

            @numba.njit
            def _traj():
                # Number of iterations before concatenating arrays
                n_iter = 1000

                # Initialize arrays for storing trajectories
                pop = np.empty((n_iter, update.shape[1]), dtype=np.int64)
                tp = np.empty(n_iter, dtype=np.float64)

                # Initialize and perform simulation
                t = time_points[0]
                population = population_0.copy()
                time_points_out = np.array([t])
                pop_out = population_0.copy().reshape((1, len(population)))
                propensities = np.zeros(update.shape[0])
                j = 0
                while t < time_points[-1]:
                    i = 0
                    while i < n_iter and t < time_points[-1]:
                        # draw the event and time step
                        event, dt = _draw(propensities, population, t)

                        # Update the population
                        population += update[event, :]
                        pop[i, :] = population

                        # Increment time
                        t += dt
                        tp[i] = t

                        # Increment indexes
                        i += 1
                        j += 1

                    # Add this subtrajectory to output
                    pop_out = np.concatenate((pop_out, pop))
                    time_points_out = np.concatenate((time_points_out, tp))

                    # Reset index
                    i = 0

                return pop_out[:j, :], time_points_out[:j]

        else:

            @numba.njit
            def _traj():
                # Initialize output
                pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int64)

                # Initialize and perform simulation
                j_time = 1
                j = 0
                t = time_points[0]
                population = population_0.copy()
                population_previous = population_0.copy()
                pop_out[0, :] = population
                propensities = np.zeros(update.shape[0])
                while j < len(time_points):
                    while t < time_points[j_time]:
                        # draw the event and time step
                        event, dt = _draw(propensities, population, t)

                        # Update the population
                        _copy_population(population_previous, population)
                        population += update[event, :]

                        # Increment time
                        t += dt

                    # Update the index (Be careful about types for Numba)
                    j = np.searchsorted((time_points > t).astype(np.int64), 1)

                    # Update the population
                    for k in np.arange(j_time, min(j, len(time_points))):
                        pop_out[k, :] = population_previous

                    # Increment index
                    j_time = j

                return pop_out, None

    else:
        if return_time_points:

            def traj():
                return _gillespie_trajectory_report_time_points(
                    propensity_func,
                    update,
                    population_0,
                    time_points,
                    _gillespie_draw,
                    args=args,
                )

        else:

            def _traj():
                return _gillespie_trajectory(
                    propensity_func,
                    update,
                    population_0,
                    time_points,
                    _gillespie_draw,
                    args=args,
                )

    # Initialize output
    pop_out = [
        np.empty((len(time_points), update.shape[1]), dtype=np.int64)
        for _ in range(size)
    ]
    t_out = [None for _ in range(size)]

    # Show progress bar
    iterator = range(size)
    if progress_bar == "notebook":
        if has_tqdm:
            iterator = tqdm.tqdm_notebook(range(size))
        else:
            warning.warn("tqdm not installed; skipping progress bar.")
    elif progress_bar:
        if has_tqdm:
            iterator = tqdm.tqdm(range(size))
        else:
            warning.warn("tqdm not installed; skipping progress bar.")

    # Perform the simulations
    for i in iterator:
        pop_out[i], t_out[i] = _traj()

    return pop_out, t_out


def _gillespie_multi_fn(args):
    """Convenient function for multithreading."""
    return _gillespie_ssa(*args)


def gillespie_ssa(
    propensity_func,
    update,
    population_0,
    time_points,
    return_time_points=False,
    size=1,
    args=(),
    n_threads=1,
    progress_bar=False,
):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.

    Parameters
    ----------
    propensity_func : function
        Function with call signature 
        `propensity_func(propensities, population, t, *args) that takes
        the current propensities and population of particle counts and 
        updates the propensities for each reaction. It does not return 
        anything.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
    size : int, default 1
        Number of trajectories to sample per thread.
    args : tuple, default ()
        The set of parameters to be passed to propensity_func.
    n_threads : int, default 1
        Number of threads to use in the calculation.
    progress_bar : str or bool, default False
        If True, use standard tqdm progress bar. If 'notebook', use
        tqdm.notebook progress bar. If False, no progress bar.

    Returns
    -------
    if `return_time_points` is False:
        sample : ndarray
            Entry i, j, k is the count of chemical species k at time
            time_points[j] for trajectory i. The shape of the array is
            (size*n_threads, num_time_points, num_chemical_species).
    if `return_time_points` is True:
        samples : list of 2d Numpy arrays
            Entry i corresponds to a trajectory. sample[i][j,k] is the 
            count of chemical species k at the jth time point for
            trajectory i.
        times : list of 1d Numpy arrays
            Entry i corresponds to a trajectory. times[i][j] is the time
            for the transition that brought the simulation to a count
            given by sanples[i][j,:].
    """
    # Check inputs
    if type(args) != tuple:
        raise RuntimeError("`args` must be a tuple, not " + str(type(args)))
    population_0 = population_0.astype(int)
    update = update.astype(int)
    time_points = np.array(time_points, dtype=float)

    if len(time_points) == 2 and not return_time_points:
        warnings.warn(
            "`return_time_points` is False, and you only have two time points inputted."
        )

    if n_threads == 1:
        pop, time = _gillespie_ssa(
            propensity_func,
            update,
            population_0,
            time_points,
            return_time_points=return_time_points,
            size=size,
            args=args,
            progress_bar=progress_bar,
        )
        if return_time_points:
            return pop, time
        else:
            return np.concatenate(pop, axis=0)
    else:
        input_args = (
            propensity_func,
            update,
            population_0,
            time_points,
            return_time_points,
            size,
            args,
            progress_bar,
        )

        with multiprocessing.Pool(n_threads) as p:
            results = p.map(_gillespie_multi_fn, [input_args] * n_threads)

        pops = [results[i][0][k] for i in range(n_threads) for k in range(size)]

        if return_time_points:
            times = [results[i][1][k] for i in range(n_threads) for k in range(size)]

            return pops, times
        else:
            return np.stack(pops, axis=0)
