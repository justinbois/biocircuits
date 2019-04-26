import numpy as np
import numba

import multiprocessing

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


def _gillespie_draw(propensity_func, population, t, args):
    """
    Draws a reaction and the time it took to do that reaction.
    """
    # Compute propensities
    props = propensity_func(population, t, *args)

    # Sum of propensities
    props_sum = _sum(props)

    # Compute time
    time = _draw_time(props_sum)

    # Draw reaction given propensities
    rxn = _sample_discrete(props, props_sum)

    return rxn, time


def _gillespie_trajectory(propensity_func, update, population_0,
                          time_points, draw_fun, args=()):

    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int64)

    # Initialize and perform simulation
    j_time = 1
    j = 0
    t = time_points[0]
    population = population_0.copy()
    pop_out[0, :] = population
    while j < len(time_points):
        while t < time_points[j_time]:
            # draw the event and time step
            event, dt = draw_fun(propensity_func, population, t, args)

            # Update the population
            population_previous = population.copy()
            population += update[event,:]

            # Increment time
            t += dt

        # Update the index (Have to be careful about types for Numba)
        j = np.searchsorted((time_points > t).astype(np.int64), 1)

        # Update the population
        for k in np.arange(j_time, min(j, len(time_points))):
            pop_out[k,:] = population_previous

        # Increment index
        j_time = j

    return pop_out


def _gillespie_ssa(propensity_func, update, population_0,
                   time_points, size=1, args=(), progress_bar=False):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.

    Parameters
    ----------
    propensity_func : function
        Function with call signature `f(population, t, *args) that takes
        the current population of particle counts and return an array of
        propensities for each reaction.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
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
    # Build trajectory function based on if propensity function is jitted
    if type(propensity_func) == numba.targets.registry.CPUDispatcher:
        @numba.njit
        def _draw(population, t):
            """
            Draws a reaction and the time it took to do that reaction.
            """
            # Compute propensities
            props = propensity_func(population, t, *args)

            # Sum of propensities
            props_sum = np.sum(props)

            # Compute time
            time = np.random.exponential(1 / props_sum)

            # Draw reaction given propensities
            rxn = _sample_discrete(props, props_sum)

            return rxn, time

        @numba.njit
        def _traj():
            # Initialize output
            pop_out = np.empty((len(time_points), update.shape[1]),
                               dtype=np.int64)

            # Initialize and perform simulation
            j_time = 1
            j = 0
            t = time_points[0]
            population = population_0.copy()
            pop_out[0, :] = population
            while j < len(time_points):
                while t < time_points[j_time]:
                    # draw the event and time step
                    event, dt = _draw(population, t)

                    # Update the population
                    population_previous = population.copy()
                    population += update[event,:]

                    # Increment time
                    t += dt

                # Update the index (Have to be careful about types for Numba)
                j = np.searchsorted((time_points > t).astype(np.int64), 1)

                # Update the population
                for k in np.arange(j_time, min(j, len(time_points))):
                    pop_out[k,:] = population_previous

                # Increment index
                j_time = j

            return pop_out
    else:
        def _traj():
            return _gillespie_trajectory(propensity_func, update,
                                         population_0, time_points,
                                         _gillespie_draw, args=args)

    # Initialize output
    pop_out = np.empty((size, len(time_points), update.shape[1]),
                       dtype=np.int64)

    # Show progress bar
    iterator = range(size)
    if progress_bar == 'notebook':
        if has_tqdm:
            iterator = tqdm.tqdm_notebook(range(size))
        else:
            warning.warn('tqdm not installed; skipping progress bar.')
    elif progress_bar:
        if has_tqdm:
            iterator = tqdm.tqdm(range(size))
        else:
            warning.warn('tqdm not installed; skipping progress bar.')

    # Perform the simulations
    for i in iterator:
        pop_out[i, :, :] = _traj()

    return pop_out


def _gillespie_multi_fn(args):
    """Convenient function for multithreading."""
    return _gillespie_ssa(*args)


def gillespie_ssa(propensity_func, update, population_0,
                  time_points, size=1, args=(), n_threads=1,
                  progress_bar=False):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.

    Parameters
    ----------
    propensity_func : function
        Function with call signature `f(population, t, *args) that takes
        the current population of particle counts and return an array of
        propensities for each reaction.
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
    sample : ndarray
        Entry i, j, k is the count of chemical species k at time
        time_points[j] for trajectory i. The shape of the array is
        (size*n_threads, num_time_points, num_chemical_species).
    """
    # Check inputs
    if type(args) != tuple:
        raise RuntimeError('`args` must be a tuple, not ' + str(type(args)))
    population_0 = population_0.astype(int)
    update = update.astype(int)

    if n_threads == 1:
        return _gillespie_ssa(propensity_func, update, population_0,
                              time_points, size=size, args=args,
                              progress_bar=progress_bar)
    else:
        input_args = (propensity_func, update, population_0,
                      time_points, size, args, progress_bar)

        with multiprocessing.Pool(n_threads) as p:
            populations = p.map(_gillespie_multi_fn, [input_args]*n_threads)

        return np.concatenate(populations)
