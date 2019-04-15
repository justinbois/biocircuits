import numpy as np

import matplotlib.streamplot

import bokeh.application
import bokeh.application.handlers
import bokeh.layouts
import bokeh.models
import bokeh.plotting

def interactive_xy_plot(base_plot, callback, slider_params=(), 
                        toggle_params=(), extra_args=()):
    """
    Create an interactive x-y plot in Bokeh.

    Parameters
    ----------
    base_plot : function
        A function to generate the initial plot that will be
        interactive. It must have call signature
        `base_plot(callback, sliders, toggles, extra_args))`, with the
        following arguments.
          callback:   A function to update the data source of the plot,
                      described as the `callback` argument below.
          sliders:    A tuple of `bokeh.models.Slider` objects.
                      Alternatively, can be any object reference-able
                      like `sliders[0].start`, `sliders[0].value`, etc.
          toggles:    A tuple of `bokeh.models.Toggle` objects.
                      Alternatively, can be any object reference-able
                      like `toggles[0].active`.
          extra_args: Tuple of any extra arguments that are passed to
                      the `callback` function.
        The `base_plot` function must return a Bokeh Figure instance and
        a Bokeh ColumnDataSource.
    callback : function
        A function that is executed to update the `ColumnDataSource` of
        the interactive plot whenever a slider or toggle are updated.
        It must have a call signature
        `callback(source, x_range, y_range, sliders, toggles, 
                  *extra_args)`.
        Here, `source` is a `ColumnDataSource`, `x_range` is the
        x_range of the plot, and `y_range` is the y_range for the plot.
        `sliders`, `toggles`, and `extra_args` are as defined above.
    slider_params : tuple of objects, default empty tuple
        Each object in the tuple is an instance of a class that has the
        following attributes.
            title : The name of the slider.
            start : The smallest value of the slider.
            end : The largest value of the slider.
            value : The starting value of the slider.
            step : The step size of the slider as it is moved.
    toggle_params : tuple of objects, default empty tuple
        Each object in the tuple is an instance of a class that has the
        following attributes.
            title : The name of the toggle.
            active : A Boolean saying whether the toggle is active.
    extra_args : tuple, default empty tuple
        Tuple of any extra arguments that are passed to the
        `callback` function.

    Returns
    -------
    output : Bokeh application
        A Bokeh application with sliders, toggles, and a plot.
    """
    def _plot_app(doc):
        # Build the initial plot and data source
        p, source = base_plot(callback, slider_params, toggle_params, extra_args)

        # Make sure axis ranges have no padding
        if type(p.x_range) == bokeh.models.ranges.Range1d:
            start, end = p.x_range.start, p.x_range.end
            p.x_range = bokeh.models.ranges.DataRange1d(p.x_range)
            p.x_range.start = start
            p.x_range.end = end
        if type(p.y_range) == bokeh.models.ranges.Range1d:
            start, end = p.y_range.start, p.y_range.end
            p.y_range = bokeh.models.ranges.DataRange1d(p.y_range)
            p.y_range.start = start
            p.y_range.end = end
        p.x_range.range_padding = 0
        p.y_range.range_padding = 0

        # Callbacks
        def _callback(attr, old, new):
            callback(source, p.x_range, p.y_range, sliders, toggles,
                     *extra_args)

        # Callback for the toggle with required call signature
        def _callback_toggle(new):
            _callback(None, None, new)

        # Set up sliders
        sliders = tuple(bokeh.models.Slider(start=param.start,
                                       end=param.end,
                                       value=param.value,
                                       step=param.step,
                                       title=param.title)
                            for param in slider_params)
        for slider in sliders:
            slider.on_change('value', _callback)

        # Set up toggles
        toggles = tuple(bokeh.models.Toggle(label=param.title) for param in toggle_params)
        for toggle in toggles:
            toggle.on_click(_callback_toggle)

        # Execute callback upon changing axis values
        p.x_range.on_change('start', _callback)
        p.x_range.on_change('end', _callback)
        p.y_range.on_change('start', _callback)
        p.y_range.on_change('end', _callback)

        # Add the plot to the app
        widgets = bokeh.layouts.widgetbox(*sliders, *toggles)
        doc.add_root(bokeh.layouts.column(widgets, p))

    handler = bokeh.application.handlers.FunctionHandler(_plot_app)
    return bokeh.application.Application(handler)


def streamplot(x, y, u, v, p=None, density=1, color=None,
               line_width=None, alpha=1, arrow_size=7, minlength=0.1,
               start_points=None, maxlength=4.0, integration_direction='both',
               x_axis_label='x', y_axis_label='y', plot_width=300,
               plot_height=260, arrow_level='underlay', **kwargs):
    """Draws streamlines of a vector flow.

    Parameters
    ----------
    *x*, *y* : 1d arrays
        an *evenly spaced* grid.
    *u*, *v* : 2d arrays
        x and y-velocities. Number of rows should match length of y, and
        the number of columns should match x.
    *p* : bokeh.plotting.Figure instance, default None
        Figure to populate with glyphs. If None, create a new figure.
    *density* : float or 2-tuple
        Controls the closeness of streamlines. When `density = 1`, the domain
        is divided into a 30x30 grid---*density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    *color* : matplotlib color code, or 2d array
        Streamline color. When given an array with the same shape as
        velocities, *color* values are converted to colors using *cmap*.
    *linewidth* : numeric or 2d array, default None
        vary linewidth when given a 2d array with the same shape as velocities. If None, scale linewidth with speed.
    *arrowsize* : float
        Factor scale arrow size.
    *minlength* : float
        Minimum length of streamline in axes coordinates.
    *start_points*: Nx2 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the ``x`` and ``y`` arrays.
    *maxlength* : float
        Maximum length of streamline in axes coordinates.
    *integration_direction* : ['forward', 'backward', 'both']
        Integrate the streamline in forward, backward or both directions.
    *x_axis_label* : str, default 'x'
        Label for x-axis. Ignored if `p` is not None.
    *y_axis_label* : str, default 'y'
        Label for y-axis. Ignored if `p` is not None.
    *plot_width* : int, default 300
        Width of plot. Ignore if `p` is not None.
    *plot_height* : int, default 260
        Width of plot. Ignore if `p` is not None.
    *arrow_level* : str
        Either 'underlay' or 'overlay'.

    Returns
    -------
    bokeh.plotting.Figure instance populated with streamplot.

    Notes
    -----
    .. Adapted from matplotlib.streamplot.streamplot.py.
    """
    if p is None:
        p = bokeh.plotting.figure(plot_width=plot_width,
                                  plot_height=plot_height,
                                  x_axis_label=x_axis_label,
                                  y_axis_label=y_axis_label,
                                  x_range=bokeh.models.Range1d(x[0], x[-1]),
                                  y_range=bokeh.models.Range1d(y[0], y[-1]),
                                  **kwargs)

    if line_width is None:
        # Compute speed
        speed = np.sqrt(u**2 + v**2)

        # Make linewidths proportional to speed, with min width 0.5 and max 3
        line_width = 0.5 + 2.5 * speed / speed.max()

    xs, ys, line_widths, arrowtails, arrowheads = _streamlines(
                x, y, u, v, density=density, line_width=line_width,
                minlength=minlength, start_points=start_points,
                maxlength=maxlength,
                integration_direction=integration_direction)

    p.multi_line(xs,
                 ys,
                 color=color,
                 line_width=line_widths,
                 line_alpha=alpha)

    for tail, head in zip(arrowtails, arrowheads):
        p.add_layout(bokeh.models.Arrow(line_alpha=0,
                                        end=bokeh.models.NormalHead(
                                            fill_color='thistle',
                                            line_alpha=0,
                                            size=7,
                                            level=arrow_level),
                                        x_start=tail[0],
                                        y_start=tail[1],
                                        x_end=head[0],
                                        y_end=head[1]))

    return p


def _streamlines(x, y, u, v, density=1, line_width=1,
               minlength=0.1, start_points=None,
               maxlength=4.0, integration_direction='both'):
    """Gives specs for streamlines of a vector flow."""
    grid = matplotlib.streamplot.Grid(x, y)
    mask = matplotlib.streamplot.StreamMask(density)
    dmap = matplotlib.streamplot.DomainMap(grid, mask)

    if integration_direction not in ['both', 'forward', 'backward']:
        errstr = ("Integration direction '%s' not recognised. "
                  "Expected 'both', 'forward' or 'backward'." %
                  integration_direction)
        raise ValueError(errstr)

    if integration_direction == 'both':
        maxlength /= 2.

    if isinstance(line_width, np.ndarray):
        if line_width.shape != grid.shape:
            raise ValueError(
            "If 'line_width' is given, must have the shape of 'Grid(x,y)'")
        line_widths = []
    else:
        line_widths = line_width


    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must be of shape 'Grid(x,y)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = matplotlib.streamplot.get_integrator(
                        u, v, dmap, minlength, maxlength,
                        integration_direction)

    trajectories = []
    if start_points is None:
        for xm, ym in matplotlib.streamplot._gen_starting_points(mask.shape):
            if mask[ym, xm] == 0:
                xg, yg = dmap.mask2grid(xm, ym)
                t = integrate(xg, yg)
                if t is not None:
                    trajectories.append(t)
    else:
        sp2 = np.asanyarray(start_points, dtype=float).copy()

        # Check if start_points are outside the data boundaries
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                    and grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError("Starting point ({}, {}) outside of data "
                                 "boundaries".format(xs, ys))

        # Convert start_points from data to array coords
        # Shift the seed points from the bottom left of the data so that
        # data2grid works properly.
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin

        for xs, ys in sp2:
            xg, yg = dmap.data2grid(xs, ys)
            t = integrate(xg, yg)
            if t is not None:
                trajectories.append(t)

    slines = []
    arrowtails = []
    arrowheads = []
    for t in trajectories:
        tgx = np.array(t[0])
        tgy = np.array(t[1])
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin

        points = np.transpose([tx, ty]).reshape(-1, 1, 2)
        slines.extend(np.hstack([points[:-1], points[1:]]))

        # Add arrows half way along each trajectory.
        s = np.cumsum(np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2))
        n = np.searchsorted(s, s[-1] / 2.)
        arrowtails.append((tx[n], ty[n]))
        arrowheads.append((np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2])))

        if isinstance(line_width, np.ndarray):
            line_widths.extend(matplotlib.streamplot.interpgrid(
                                                line_width, tgx, tgy)[:-1])

    xs = [s[:,0] for s in slines]
    ys = [s[:,1] for s in slines]

    return xs, ys, line_widths, arrowtails, arrowheads


def mpl_cmap_to_color_mapper(cmap):
    """
    Convert a Matplotlib colormap to a bokeh.models.LinearColorMapper
    instance.

    Parameters
    ----------
    cmap : str
        A string giving the name of the color map.

    Returns
    -------
    output : bokeh.models.LinearColorMapper instance
        A linear color_mapper with 25 gradations.

    Notes
    -----
    .. See https://matplotlib.org/examples/color/colormaps_reference.html
       for available Matplotlib colormaps.
    """
    cm = mpl_get_cmap(cmap)
    palette = [rgb_frac_to_hex(cm(i)[:3]) for i in range(256)]
    return bokeh.models.LinearColorMapper(palette=palette)

