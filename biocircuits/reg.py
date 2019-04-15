def aa_and(x, y, nx, ny):
    """Dimensionless production rate for a gene regulated by two 
    activators with AND logic in the absence of leakage.

    Parameters
    ----------
    x : float or NumPy array
        Concentration of first activator.
    y : float or NumPy array
        Concentration of second activator.
    nx : float
        Hill coefficient for first activator.
    ny : float
        Hill coefficient for second activator.

    Returns
    -------
    output : NumPy array or float
        x**nx * y**ny / (1 + x**nx * y**ny)
    """
    return x**nx * y**ny / (1 + x**nx * y**ny)


def aa_or(x, y, nx, ny):
    """Dimensionless production rate for a gene regulated by two 
    activators with OR logic in the absence of leakage.

    Parameters
    ----------
    x : float or NumPy array
        Concentration of first activator.
    y : float or NumPy array
        Concentration of second activator.
    nx : float
        Hill coefficient for first activator.
    ny : float
        Hill coefficient for second activator.

    Returns
    -------
    output : NumPy array or float
        (x**nx + y**ny) / (1 + x**nx + y**ny)
    """
    return (x**nx + y**ny) / (1 + x**nx + y**ny)


def rr_and(x, y, nx, ny):
    """Dimensionless production rate for a gene regulated by two 
    repressors with AND logic in the absence of leakage.

    Parameters
    ----------
    x : float or NumPy array
        Concentration of first repressor.
    y : float or NumPy array
        Concentration of second repressor.
    nx : float
        Hill coefficient for first repressor.
    ny : float
        Hill coefficient for second repressor.

    Returns
    -------
    output : NumPy array or float
        1 / (1 + x**nx) / (1 + y**ny)
    """
    return 1 / (1 + x**nx) / (1 + y**ny)


def rr_or(x, y, nx, ny):
    """Dimensionless production rate for a gene regulated by two 
    repressors with OR logic in the absence of leakage.

    Parameters
    ----------
    x : float or NumPy array
        Concentration of first repressor.
    y : float or NumPy array
        Concentration of second repressor.
    nx : float
        Hill coefficient for first repressor.
    ny : float
        Hill coefficient for second repressor.

    Returns
    -------
    output : NumPy array or float
        (1 + x**nx + y**ny) / (1 + x**nx) / (1 + y**ny)
    """
    return (1 + x**nx + y**ny) / (1 + x**nx) / (1 + y**ny)


def ar_and(x, y, nx, ny):
    """Dimensionless production rate for a gene regulated by one 
    activator and one repressor with AND logic in the absence of 
    leakage.

    Parameters
    ----------
    x : float or NumPy array
        Concentration of activator.
    y : float or NumPy array
        Concentration of repressor.
    nx : float
        Hill coefficient for activator.
    ny : float
        Hill coefficient for repressor.

    Returns
    -------
    output : NumPy array or float
        x**nx / (1 + x**nx + y**ny)
    """
    return x**nx / (1 + x**nx + y**ny)


def ar_or(x, y, nx, ny):
    """Dimensionless production rate for a gene regulated by one 
    activator and one repressor with OR logic in the absence of 
    leakage.

    Parameters
    ----------
    x : float or NumPy array
        Concentration of activator.
    y : float or NumPy array
        Concentration of repressor.
    nx : float
        Hill coefficient for activator.
    ny : float
        Hill coefficient for repressor.

    Returns
    -------
    output : NumPy array or float
        (1 + x**nx) / (1 + x**nx + y**ny)
    """
    return (1 + x**nx) / (1 + x**nx + y**ny)