import warnings

import numpy as np

import matplotlib.streamplot

import bokeh.application
import bokeh.application.handlers
import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting

import colorcet

from . import utils


def _ecdf_vals(data, formal=False, complementary=False):
    """Get x, y, values of an ECDF for plotting.
    Parameters
    ----------
    data : ndarray
        One dimensional Numpy array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    complementary : bool
        If True, return values for ECCDF.
    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    if formal:
        x, y = _to_formal(x, y)
        if complementary:
            y = 1 - y
    elif complementary:
        y = 1 - y + 1 / len(y)

    return x, y


def _to_formal(x, y):
    """Convert to formal ECDF."""
    # Set up output arrays
    x_formal = np.empty(2 * len(x))
    y_formal = np.empty(2 * len(x))

    # y-values for steps
    y_formal[0] = 0
    y_formal[1::2] = y
    y_formal[2::2] = y[:-1]

    # x- values for steps
    x_formal[::2] = x
    x_formal[1::2] = x

    return x_formal, y_formal


def ecdf(
    data=None,
    conf_int=False,
    ptiles=[2.5, 97.5],
    n_bs_reps=1000,
    fill_color="lightgray",
    fill_alpha=1,
    p=None,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    plot_height=300,
    plot_width=450,
    formal=False,
    complementary=False,
    x_axis_type="linear",
    y_axis_type="linear",
    **kwargs,
):
    """
    Create a plot of an ECDF. DEPRECATED. Use iqplot.ecdf().

    Parameters
    ----------
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    conf_int : bool, default False
        If True, display a confidence interval on the ECDF.
    ptiles : list, default [2.5, 97.5]
        The percentiles to use for the confidence interval. Ignored it
        `conf_int` is False.
    n_bs_reps : int, default 1000
        Number of bootstrap replicates to do to compute confidence
        interval. Ignored if `conf_int` is False.
    fill_color : str, default 'lightgray'
        Color of the confidence interbal. Ignored if `conf_int` is
        False.
    fill_alpha : float, default 1
        Opacity of confidence interval. Ignored if `conf_int` is False.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored if `p` is not None.
    y_axis_label : str, default 'ECDF' or 'ECCDF'
        Label for the y-axis. Ignored if `p` is not None.
    title : str, default None
        Title of the plot. Ignored if `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored if `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored if `p` is not None.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots.
    complementary : bool, default False
        If True, plot the empirical complementary cumulative
        distribution functon.
    x_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    y_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    kwargs
        Any kwargs to be passed to either p.circle or p.line, for
        `formal` being False or True, respectively.
    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    warnings.warn(
        "`biocircuits.ecdf()` is deprecated. Used `iqplot.ecdf()` instead.",
        DeprecationWarning,
    )

    # Check data to make sure legit
    data = utils._convert_data(data)

    # Get y-axis label
    if p is None and y_axis_label is None:
        if complementary:
            y_axis_label = "ECCDF"
        else:
            y_axis_label = "ECDF"

    # Data points on ECDF
    x, y = _ecdf_vals(data, formal, complementary)

    # Instantiate Bokeh plot if not already passed in
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height,
            plot_width=plot_width,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            title=title,
        )

    # Do bootstrap replicates
    if conf_int:
        x_plot = np.sort(np.unique(x))
        bs_reps = np.array(
            [
                _ecdf_arbitrary_points(np.random.choice(data, size=len(data)), x_plot)
                for _ in range(n_bs_reps)
            ]
        )

        # Compute the confidence intervals
        ecdf_low, ecdf_high = np.percentile(np.array(bs_reps), ptiles, axis=0)

        # Make them formal
        _, ecdf_low = _to_formal(x=x_plot, y=ecdf_low)
        x_plot, ecdf_high = _to_formal(x=x_plot, y=ecdf_high)

        p = fill_between(
            x1=x_plot,
            y1=ecdf_low,
            x2=x_plot,
            y2=ecdf_high,
            fill_color=fill_color,
            show_line=False,
            p=p,
        )

    if formal:
        # Line of steps
        p.line(x, y, **kwargs)

        # Rays for ends
        if complementary:
            p.ray(x[0], 1, None, np.pi, **kwargs)
            p.ray(x[-1], 0, None, 0, **kwargs)
        else:
            p.ray(x[0], 0, None, np.pi, **kwargs)
            p.ray(x[-1], 1, None, 0, **kwargs)
    else:
        p.circle(x, y, **kwargs)

    return p


def interactive_xy_plot(
    base_plot, callback, slider_params=(), toggle_params=(), extra_args=()
):
    """
    Create an interactive x-y plot in Bokeh. DEPRECATED.

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
    warnings.warn(
        DeprecationWarning,
        "`interactive_xy_plot() is deprecated. Either custom build an interactive plot with base Bokeh or use Panel.",
    )

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
            callback(source, p.x_range, p.y_range, sliders, toggles, *extra_args)

        # Callback for the toggle with required call signature
        def _callback_toggle(new):
            _callback(None, None, new)

        # Set up sliders
        sliders = tuple(
            bokeh.models.Slider(
                start=param.start,
                end=param.end,
                value=param.value,
                step=param.step,
                title=param.title,
            )
            for param in slider_params
        )
        for slider in sliders:
            slider.on_change("value", _callback)

        # Set up toggles
        toggles = tuple(
            bokeh.models.Toggle(label=param.title) for param in toggle_params
        )
        for toggle in toggles:
            toggle.on_click(_callback_toggle)

        # Execute callback upon changing axis values
        p.x_range.on_change("start", _callback)
        p.x_range.on_change("end", _callback)
        p.y_range.on_change("start", _callback)
        p.y_range.on_change("end", _callback)

        # Add the plot to the app
        widgets = bokeh.layouts.column(*sliders, *toggles)
        doc.add_root(bokeh.layouts.column(widgets, p))

    handler = bokeh.application.handlers.FunctionHandler(_plot_app)
    return bokeh.application.Application(handler)


def xyt_plot(
    x,
    y,
    t,
    glyph="line",
    legend_names=None,
    legend_location="top_right",
    palette=None,
    glyph_kwargs={},
    time_slider_title="time",
    **kwargs,
):
    """Create an interactive plot some the output of `rd_solve()`.

    Parameters
    ----------
    x: 1D Numpy array
        x-values for plot.
    y : tuple of 2D Numpy arrays
        y-values for plot. `y[i][j, k]` is the y-value for curve i at
        time point j and position x[k].
    t : Numpy array
        y-values for plot.
    glyph : str, either "line" or "circle"
        What kind of glyph to use in plot.
    legend_names : list or None, default None
        Names for legend in plot.
    legend_location : str, default 'top_right'
        Position of legend using Bokeh's specifications for legend
        position. Ignored if `legend_names` is None.
    palette : List of colors default bokeh.palettes.d3['Category10'][10]
        Color palette to use for curves.
    kwargs :
        All other kwargs are passed to bokeh.plotting.figure() in
        creating the figure.

    Returns
    -------
    output
        An interactive plot with a time slider.
    """
    if palette is None:
        palette = colorcet.b_glasbey_category10

    if glyph not in ["line", "circle"]:
        raise RuntimeError("Only 'line' or 'circle' glyphs allowed.")

    # Set defaults
    if glyph == "line" and "line_width" not in glyph_kwargs:
        glyph_kwargs["line_width"] = 2
    if "x_axis_label" not in kwargs:
        kwargs["x_axis_label"] = "x"
    if "y_axis_label" not in kwargs:
        kwargs["y_axis_label"] = "y"
    if "frame_height" not in kwargs and "plot_height" not in kwargs:
        kwargs["frame_height"] = 275
    if "frame_width" not in kwargs and "plot_width" not in kwargs:
        kwargs["frame_width"] = 450
    if "y_range" not in kwargs:
        total_range_min = np.concatenate(y).min()
        total_range_max = np.concatenate(y).max()
        total_range = total_range_max - total_range_min
        kwargs["y_range"] = [
            total_range_min - total_range * 0.02,
            total_range_max + total_range * 0.02,
        ]
    if "x_range" not in kwargs:
        kwargs["x_range"] = [x.min(), x.max()]

    if "color" not in glyph_kwargs and len(y) > len(palette):
        raise RuntimeError("Not enough colors in palette for curves.")

    if legend_names is None:
        legend_names = [None for _ in range(len(y))]
    elif len(legend_names) != len(y):
        raise RuntimeError("len(legend_names) must equal len(y).")

    t_slider = bokeh.models.Slider(
        title=time_slider_title,
        name="t",
        start=t[0],
        end=t[-1],
        value=t[0],
        step=np.diff(t).mean(),
    )

    # Build data sources, source has all of the data
    data_dict = {"y_" + str(j): [] for j in range(len(y))}
    data_dict["i"] = []
    for i in range(len(t)):
        data_dict["i"] += [i] * len(x)
        for j, c in enumerate(y):
            data_dict["y_" + str(j)] += list(c[i, :])
    source = bokeh.models.ColumnDataSource(data_dict)

    # source_plot has only the data that will be displayed at a given time
    plot_dict = {
        "y_" + str(j): data_dict["y_" + str(j)][: len(x)] for j in range(len(y))
    }
    plot_dict["x"] = x
    source_plot = bokeh.models.ColumnDataSource(plot_dict)

    # The time points, need to have around
    source_t = bokeh.models.ColumnDataSource(dict(t=t))

    # Make the plot
    p = bokeh.plotting.figure(**kwargs)

    click_policy = False
    args = {key: val for key, val in glyph_kwargs.items()}
    args["source"] = source_plot
    for i, leg in enumerate(legend_names):
        if "color" not in glyph_kwargs:
            args["color"] = palette[i]

        if leg is not None:
            args["legend_label"] = str(leg)
            click_policy = True

        if glyph == "line":
            p.line("x", "y_" + str(i), **args)
        elif glyph == "circle":
            p.circle("x", "y_" + str(i), **args)

    if click_policy:
        p.legend.click_policy = "hide"
        p.legend.location = legend_location

    # Callback
    js_code = """
function sortedIndex(array, value) {
    let low = 0,
        high = array.length;

    while (low < high) {
        let mid = (low + high) >>> 1;
        if (array[mid] < value) low = mid + 1;
        else high = mid;
    }
    return low;
}

let x = source_plot.data['x'];
let x_len = x.length;
let t = source_t.data['t'];

let i = sortedIndex(t, cb_obj.value);
"""

    for var_name in ["y_" + str(j) for j in range(len(y))]:
        js_code += f"""let {var_name} = source_plot.data['{var_name}'];
let {var_name}_source = source.data['{var_name}'];
for (let k = 0; k < x_len; k++) {var_name}[k] = {var_name}_source[x_len * i + k];

"""

    js_code += "source_plot.change.emit();\n"

    callback = bokeh.models.CustomJS(
        args=dict(source=source, source_t=source_t, source_plot=source_plot),
        code=js_code,
    )

    t_slider.js_on_change("value", callback)

    return bokeh.layouts.column(
        bokeh.layouts.row(bokeh.models.Spacer(width=10), t_slider), p
    )


def phase_portrait(
    du_dt,
    dv_dt,
    u_range,
    v_range,
    args_u=(),
    args_v=(),
    log=False,
    p=None,
    zoomable=False,
    **kwargs,
):
    """
    Plots the phase portrait for a 2D dynamical system in the u-v plane.

    Parameters
    ----------
    du_dt : function
        A function to compute the right hand side of du/dt. Must have
        call signature `du_dt(u, v, *args_u)`. Note that there can be
        no explicit time dependence.
    dv_dt : function
        A function to compute the right hand side of dv/dt. Must have
        call signature `dv_dt(u, v, *args_v)`. Note that there can be
        no explicit time dependence.
    u_range : array_like, shape (2,)
        Minimum and maximum values of u to consider.
    v_range : array_like, shape (2,)
        Minimum and maximum values of v to consider.
    args_u : tuple, default ()
        Tuple of extra arguments to be passed to `du_dt`.
    args_v : tuple, default ()
        Tuple of extra arguments to be passed to `dv_dt`.
    log : bool, default False
        If True, plot u and v on a logarithmic scale.
    p : bokeh.plotting.Figure instance, default None
        Figure to use for the phase portrait. If None, a new one is
        created according to `streamplot()`.
    zoomable : bool, default False
        If True, generates a zoomable plot where the streamlines will be
        redrawn to correspond for the level so zoom. This requires
        Python to be running for the plot to render.
    kwargs :
        All other kwargs are passed to `streamplot`.

    Returns
    -------
    output : bokeh.plotting.Figure instance populated with streamplot
    """
    if zoomable:
        _kwargs = {f"_{key}": val for key, val in kwargs.items()}
        return _zoomable_phase_portrait(
            du_dt, dv_dt, u_range, v_range, args_u, args_v, log, p, **_kwargs
        )
    else:
        return _phase_portrait(
            du_dt, dv_dt, u_range, v_range, args_u, args_v, log, p, **kwargs
        )


def _phase_portrait(du_dt, dv_dt, u_range, v_range, args_u, args_v, log, p, **kwargs):
    if log:
        # Set up u,v space
        log_u = np.linspace(np.log10(u_range[0]), np.log10(u_range[1]), 100)
        log_v = np.linspace(np.log10(v_range[0]), np.log10(v_range[1]), 100)
        log_uu, log_vv = np.meshgrid(log_u, log_v)

        # Compute derivatives
        log_u_vel = du_dt(10 ** log_uu, 10 ** log_vv, *args_u) / 10 ** log_uu
        log_v_vel = dv_dt(10 ** log_uu, 10 ** log_vv, *args_v) / 10 ** log_vv

        # Make stream plot
        return streamplot(log_u, log_v, log_u_vel, log_v_vel, p=p, **kwargs)
    else:
        # Set up u,v space
        u = np.linspace(u_range[0], u_range[1], 100)
        v = np.linspace(v_range[0], v_range[1], 100)
        uu, vv = np.meshgrid(u, v)

        # Compute derivatives
        u_vel = du_dt(uu, vv, *args_u)
        v_vel = dv_dt(uu, vv, *args_v)

        # Make stream plot
        return streamplot(u, v, u_vel, v_vel, p=p, **kwargs)


def streamplot(
    x,
    y,
    u,
    v,
    p=None,
    density=1,
    color="#1f77b4",
    line_width=None,
    alpha=1,
    arrow_size=7,
    min_length=0.1,
    start_points=None,
    max_length=4.0,
    integration_direction="both",
    arrow_level="underlay",
    **kwargs,
):
    """Draws streamlines of a vector field.

    Parameters
    ----------
    x, y : 1d arrays
        an evenly spaced grid.
    u, v : 2d arrays
        x and y-velocities. Number of rows should match length of y, and
        the number of columns should match x.
    p : bokeh.plotting.Figure instance, default None
        Figure to populate with glyphs. If None, create a new figure.
    density : float or 2-tuple
        Controls the closeness of streamlines. When `density = 1`, the domain
        is divided into a 30x30 grid---density linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    color : str or 2d array, default '#1f77b4' (Bokeh default color)
        Streamline color. When given an array with the same shape as
        velocities, color values are converted to colors using cmap.
    line_width : numeric or 2d array, default None
        vary linewidth when given a 2d array with the same shape as velocities. If None, scale linewidth with speed.
    arrow_size : float
        Factor scale arrow size.
    min_length : float
        Minimum length of streamline in axes coordinates.
    start_points: Nx2 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the ``x`` and ``y`` arrays.
    max_length : float
        Maximum length of streamline in axes coordinates.
    integration_direction : ['forward', 'backward', 'both']
        Integrate the streamline in forward, backward or both directions.
    arrow_level : str
        Either 'underlay' or 'overlay'.
    kwargs :
        All other kwargs are passed to bokeh.plotting.figure() when
        generating the figure.

    Returns
    -------
    bokeh.plotting.Figure instance populated with streamplot.

    Notes
    -----
    .. Adapted from matplotlib.streamplot.streamplot.py.
    """
    if p is None:
        p = _baseplot(p, **kwargs)

    # Ensure plot fits stream lines
    p.x_range = bokeh.models.Range1d(x[0], x[-1])
    p.y_range = bokeh.models.Range1d(y[0], y[-1])

    if line_width is None:
        # Compute speed
        speed = np.sqrt(u ** 2 + v ** 2)

        # Make linewidths proportional to speed, with min width 0.5 and max 3
        line_width = 0.5 + 2.5 * speed / speed.max()

    xs, ys, line_widths, arrowtails, arrowheads = _streamlines(
        x,
        y,
        u,
        v,
        density=density,
        line_width=line_width,
        min_length=min_length,
        start_points=start_points,
        max_length=max_length,
        integration_direction=integration_direction,
    )

    def _draw_arrows():
        for tail, head in zip(arrowtails, arrowheads):
            p.add_layout(
                bokeh.models.Arrow(
                    line_alpha=0,
                    end=bokeh.models.NormalHead(fill_color=color, line_alpha=0, size=7),
                    x_start=tail[0],
                    y_start=tail[1],
                    x_end=head[0],
                    y_end=head[1],
                )
            )

    if arrow_level == "underlay":
        _draw_arrows()
        p.multi_line(xs, ys, color=color, line_width=line_widths, line_alpha=alpha)
    else:
        p.multi_line(xs, ys, color=color, line_width=line_widths, line_alpha=alpha)
        _draw_arrows()

    return p


def _baseplot(p, **kwargs):
    if p is None:
        if "x_axis_label" not in kwargs:
            kwargs["x_axis_label"] = "x"
        if "y_axis_label" not in kwargs:
            kwargs["y_axis_label"] = "y"
        if "x_range" in kwargs or "y_range" in kwargs:
            raise RuntimeError("Cannot specify ranges in kwargs.")
        if (
            "plot_height" not in kwargs
            or "frame_height" not in kwargs
            or "height" not in kwargs
        ):
            kwargs["frame_height"] = 260
        if (
            "plot_width" not in kwargs
            or "frame_width" not in kwargs
            or "width" not in kwargs
        ):
            kwargs["frame_width"] = 260
        if "tools" not in kwargs:
            tools = "box_zoom,save,reset"

        p = bokeh.plotting.figure(**kwargs)

    return p


def _give_back_local_params(
    du_dt,
    dv_dt,
    u_range,
    v_range,
    args_u,
    args_v,
    log,
    p,
    density,
    color,
    line_width,
    alpha,
    arrow_size,
    min_length,
    start_points,
    max_length,
    integration_direction,
    arrow_level,
):

    return (
        du_dt,
        dv_dt,
        u_range,
        v_range,
        args_u,
        args_v,
        log,
        p,
        density,
        color,
        line_width,
        alpha,
        arrow_size,
        min_length,
        start_points,
        max_length,
        integration_direction,
        arrow_level,
    )


def _zoomable_phase_portrait(
    _du_dt,
    _dv_dt,
    _u_range,
    _v_range,
    _args_u,
    _args_v,
    _log,
    _p,
    _density=1,
    _color="#1f77b4",
    _line_width=None,
    _alpha=1,
    _arrow_size=7,
    _min_length=0.1,
    _start_points=None,
    _max_length=4.0,
    _integration_direction="both",
    _arrow_level="underlay",
    **kwargs,
):
    """Draws streamlines of a vector flow.

    Parameters
    ----------
    x, y : 1d arrays
        an evenly spaced grid.
    u, v : 2d arrays
        x and y-velocities. Number of rows should match length of y, and
        the number of columns should match x.
    p : bokeh.plotting.Figure instance, default None
        Figure to populate with glyphs. If None, create a new figure.
    density : float or 2-tuple
        Controls the closeness of streamlines. When `density = 1`, the domain
        is divided into a 30x30 grid---density linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    color : str or 2d array, default '#1f77b4' (Bokeh default color)
        Streamline color. When given an array with the same shape as
        velocities, color values are converted to colors using cmap.
    line_width : numeric or 2d array, default None
        vary linewidth when given a 2d array with the same shape as velocities. If None, scale linewidth with speed.
    arrow_size : float
        Factor scale arrow size.
    min_length : float
        Minimum length of streamline in axes coordinates.
    start_points: Nx2 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the ``x`` and ``y`` arrays.
    max_length : float
        Maximum length of streamline in axes coordinates.
    integration_direction : ['forward', 'backward', 'both']
        Integrate the streamline in forward, backward or both directions.
    arrow_level : str
        Either 'underlay' or 'overlay'.
    kwargs :
        All other kwargs are passed to bokeh.plotting.figure() when
        generating the figure.

    Returns
    -------
    bokeh.plotting.Figure instance populated with streamplot.

    Notes
    -----
    .. Adapted from matplotlib.streamplot.streamplot.py.
    """

    def _plot_app(doc):
        (
            du_dt,
            dv_dt,
            u_range,
            v_range,
            args_u,
            args_v,
            log,
            p,
            density,
            color,
            line_width,
            alpha,
            arrow_size,
            min_length,
            start_points,
            max_length,
            integration_direction,
            arrow_level,
        ) = _give_back_local_params(
            _du_dt,
            _dv_dt,
            _u_range,
            _v_range,
            _args_u,
            _args_v,
            _log,
            _p,
            _density,
            _color,
            _line_width,
            _alpha,
            _arrow_size,
            _min_length,
            _start_points,
            _max_length,
            _integration_direction,
            _arrow_level,
        )
        p = _baseplot(p, **kwargs)

        # Ensure plot fits stream lines
        if log:
            p.x_range = bokeh.models.Range1d(
                np.log10(u_range[0]), np.log10(u_range[-1])
            )
            p.y_range = bokeh.models.Range1d(
                np.log10(v_range[0]), np.log10(v_range[-1])
            )
        else:
            p.x_range = bokeh.models.Range1d(u_range[0], u_range[-1])
            p.y_range = bokeh.models.Range1d(v_range[0], v_range[-1])

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

        # Only allow certain tools
        p.tools = [
            bokeh.models.BoxZoomTool(),
            bokeh.models.ResetTool(),
            bokeh.models.SaveTool(),
        ]

        if log:
            # Set up u,v space
            u = np.linspace(np.log10(u_range[0]), np.log10(u_range[1]), 100)
            v = np.linspace(np.log10(v_range[0]), np.log10(v_range[1]), 100)
            log_uu, log_vv = np.meshgrid(u, v)

            # Compute derivatives
            u_vel = du_dt(10 ** log_uu, 10 ** log_vv, *args_u) / 10 ** log_uu
            v_vel = dv_dt(10 ** log_uu, 10 ** log_vv, *args_v) / 10 ** log_vv
        else:
            # Set up u,v space
            u = np.linspace(u_range[0], u_range[1], 100)
            v = np.linspace(v_range[0], v_range[1], 100)
            uu, vv = np.meshgrid(u, v)

            # Compute derivatives
            u_vel = du_dt(uu, vv, *args_u)
            v_vel = dv_dt(uu, vv, *args_v)

        if line_width is None:
            # Compute speed
            speed = np.sqrt(u_vel ** 2 + v_vel ** 2)

            # Make linewidths proportional to speed, with min width 0.5 and max 3
            line_width_plot = 0.5 + 2.5 * speed / speed.max()
        else:
            line_width_plot = line_width

        xs, ys, line_widths, arrowtails, arrowheads = _streamlines(
            u,
            v,
            u_vel,
            v_vel,
            density=density,
            line_width=line_width_plot,
            min_length=min_length,
            start_points=start_points,
            max_length=max_length,
            integration_direction=integration_direction,
        )

        line_source = bokeh.models.ColumnDataSource(
            dict(xs=xs, ys=ys, line_width=line_widths)
        )

        arrows = [
            bokeh.models.Arrow(
                line_alpha=0,
                end=bokeh.models.NormalHead(fill_color=color, line_alpha=0, size=7),
                x_start=tail[0],
                y_start=tail[1],
                x_end=head[0],
                y_end=head[1],
            )
            for head, tail in zip(arrowheads, arrowtails)
        ]

        if arrow_level == "underlay":
            for arrow in arrows:
                p.add_layout(arrow)

            p.multi_line(
                source=line_source,
                xs="xs",
                ys="ys",
                line_color=color,
                line_width="line_width",
                line_alpha=alpha,
            )
        else:
            p.multi_line(
                source=line_source,
                xs="xs",
                ys="ys",
                line_color=color,
                line_width="line_width",
                line_alpha=alpha,
            )

            for arrow in arrows:
                p.add_layout(arrow)

        def _callback(attr, old, new):
            # Set up u,v space
            u = np.linspace(p.x_range.start, p.x_range.end, 100)
            v = np.linspace(p.y_range.start, p.y_range.end, 100)
            uu, vv = np.meshgrid(u, v)

            if log:
                # Compute derivatives
                u_vel = du_dt(10 ** uu, 10 ** vv, *args_u) / 10 ** uu
                v_vel = dv_dt(10 ** uu, 10 ** vv, *args_v) / 10 ** vv
            else:
                # Compute derivatives
                u_vel = du_dt(uu, vv, *args_u)
                v_vel = dv_dt(uu, vv, *args_v)

            if line_width is None:
                # Compute speed
                speed = np.sqrt(u_vel ** 2 + v_vel ** 2)

                # Make linewidths proportional to speed, with min width 0.5 and max 3
                line_width_plot = 0.5 + 2.5 * speed / speed.max()
            else:
                line_width_plot = line_width

            xs, ys, line_widths, arrowtails, arrowheads = _streamlines(
                u,
                v,
                u_vel,
                v_vel,
                density=density,
                line_width=line_width_plot,
                min_length=min_length,
                start_points=start_points,
                max_length=max_length,
                integration_direction=integration_direction,
            )

            line_source.data = dict(xs=xs, ys=ys, line_width=line_widths)

            if len(arrowtails) < len(arrows):
                for i, (head, tail) in enumerate(zip(arrowheads, arrowtails)):
                    arrows[i].x_start = tail[0]
                    arrows[i].y_start = tail[1]
                    arrows[i].x_end = head[0]
                    arrows[i].y_end = head[1]
                for i in range(len(arrowtails), len(arrows)):
                    arrows[i].visible = False
            else:
                for i, (head, tail) in enumerate(
                    zip(arrowheads[: len(arrows)], arrowtails[: len(arrows)])
                ):
                    arrows[i].x_start = tail[0]
                    arrows[i].y_start = tail[1]
                    arrows[i].x_end = head[0]
                    arrows[i].y_end = head[1]
                    arrows[i].visible = True

                for head, tail in zip(
                    arrowheads[len(arrows) :], arrowtails[len(arrows) :]
                ):
                    arrows.append(
                        bokeh.models.Arrow(
                            line_alpha=0,
                            end=bokeh.models.NormalHead(
                                fill_color=color, line_alpha=0, size=7,
                            ),
                            x_start=tail[0],
                            y_start=tail[1],
                            x_end=head[0],
                            y_end=head[1],
                        )
                    )

                    p.add_layout(arrows[-1])

        # Execute callback upon changing axis values
        p.x_range.on_change("start", _callback)
        p.x_range.on_change("end", _callback)
        p.y_range.on_change("start", _callback)
        p.y_range.on_change("end", _callback)

        # Add the plot to the app
        doc.add_root(p)

    handler = bokeh.application.handlers.FunctionHandler(_plot_app)
    return bokeh.application.Application(handler)


def _streamlines(
    x,
    y,
    u,
    v,
    density=1,
    line_width=1,
    min_length=0.1,
    start_points=None,
    max_length=4.0,
    integration_direction="both",
):
    """Gives specs for streamlines of a vector flow."""
    grid = matplotlib.streamplot.Grid(x, y)
    mask = matplotlib.streamplot.StreamMask(density)
    dmap = matplotlib.streamplot.DomainMap(grid, mask)

    if integration_direction not in ["both", "forward", "backward"]:
        errstr = (
            "Integration direction '%s' not recognised. "
            "Expected 'both', 'forward' or 'backward'." % integration_direction
        )
        raise ValueError(errstr)

    if integration_direction == "both":
        max_length /= 2.0

    if isinstance(line_width, np.ndarray):
        if line_width.shape != grid.shape:
            raise ValueError(
                "If 'line_width' is given, must have the shape of 'Grid(x,y)'"
            )
        line_widths = []
    else:
        line_widths = line_width

    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must be of shape 'Grid(x,y)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = matplotlib.streamplot.get_integrator(
        u, v, dmap, min_length, max_length, integration_direction
    )

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
            if not (
                grid.x_origin <= xs <= grid.x_origin + grid.width
                and grid.y_origin <= ys <= grid.y_origin + grid.height
            ):
                raise ValueError(
                    "Starting point ({}, {}) outside of data "
                    "boundaries".format(xs, ys)
                )

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
        n = np.searchsorted(s, s[-1] / 2.0)
        arrowtails.append((tx[n], ty[n]))
        arrowheads.append((np.mean(tx[n : n + 2]), np.mean(ty[n : n + 2])))

        if isinstance(line_width, np.ndarray):
            line_widths.extend(
                matplotlib.streamplot.interpgrid(line_width, tgx, tgy)[:-1]
            )

    xs = [s[:, 0] for s in slines]
    ys = [s[:, 1] for s in slines]

    return xs, ys, line_widths, arrowtails, arrowheads


def imshow(
    im,
    color_mapper=None,
    frame_height=400,
    frame_width=None,
    length_units="pixels",
    interpixel_distance=1.0,
    x_range=None,
    y_range=None,
    colorbar=False,
    no_ticks=False,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    flip=True,
    return_im=False,
    saturate_channels=True,
    min_intensity=None,
    max_intensity=None,
    display_clicks=False,
):
    """
    Display an image in a Bokeh figure.

    Parameters
    ----------
    im : Numpy array
        If 2D, intensity image to be displayed. If 3D, first two
        dimensions are pixel values. Last dimension can be of length
        1, 2, or 3, which specify colors.
    color_mapper : str or bokeh.models.LinearColorMapper, default None
        If `im` is an intensity image, `color_mapper` is a mapping of
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `color_mapper` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    plot_height : int
        Height of the plot in pixels. The width is scaled so that the
        x and y distance between pixels is the same.
    plot_width : int or None (default)
        If None, the width is scaled so that the x and y distance
        between pixels is approximately the same. Otherwise, the width
        of the plot in pixels.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    x_range : bokeh.models.Range1d instance, default None
        Range of x-axis. If None, determined automatically.
    y_range : bokeh.models.Range1d instance, default None
        Range of y-axis. If None, determined automatically.
    colorbar : bool, default False
        If True, include a colorbar.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    x_axis_label : str, default None
        Label for the x-axis. If None, labeled with `length_units`.
    y_axis_label : str, default None
        Label for the y-axis. If None, labeled with `length_units`.
    title : str, default None
        The title of the plot.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
    return_im : bool, default False
        If True, return the GlyphRenderer instance of the image being
        displayed.
    saturate_channels : bool, default True
        If True, each of the channels have their displayed pixel values
        extended to range from 0 to 255 to show the full dynamic range.
    min_intensity : int or float, default None
        Minimum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    max_intensity : int or float, default None
        Maximum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    display_clicks : bool, default False
        If True, display clicks to the right of the plot using
        JavaScript. The clicks are not recorded nor stored, just
        printed.

    Returns
    -------
    p : bokeh.plotting.figure instance
        Bokeh plot with image displayed.
    im : bokeh.models.renderers.GlyphRenderer instance (optional)
        The GlyphRenderer instance of the image being displayed. This is
        only returned if `return_im` is True.
    """

    # If a single channel in 3D image, flatten and check shape
    if im.ndim == 3:
        if im.shape[2] == 1:
            im = im[:, :, 0]
        elif im.shape[2] not in [2, 3]:
            raise RuntimeError("Can only display 1, 2, or 3 channels.")

    # If binary image, make sure it's int
    if im.dtype == bool:
        im = im.astype(np.uint8)

    # Get color mapper
    if im.ndim == 2:
        if color_mapper is None:
            color_mapper = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))
        elif type(color_mapper) == str and color_mapper.lower() in ["rgb", "cmy"]:
            raise RuntimeError("Cannot use rgb or cmy colormap for intensity image.")
        if min_intensity is None:
            color_mapper.low = im.min()
        else:
            color_mapper.low = min_intensity
        if max_intensity is None:
            color_mapper.high = im.max()
        else:
            color_mapper.high = max_intensity
    elif im.ndim == 3:
        if color_mapper is None or color_mapper.lower() == "cmy":
            im = im_merge(
                *np.rollaxis(im, 2),
                cmy=True,
                im_0_min=min_intensity,
                im_1_min=min_intensity,
                im_2_min=min_intensity,
                im_0_max=max_intensity,
                im_1_max=max_intensity,
                im_2_max=max_intensity,
            )
        elif color_mapper.lower() == "rgb":
            im = im_merge(
                *np.rollaxis(im, 2),
                cmy=False,
                im_0_min=min_intensity,
                im_1_min=min_intensity,
                im_2_min=min_intensity,
                im_0_max=max_intensity,
                im_1_max=max_intensity,
                im_2_max=max_intensity,
            )
        else:
            raise RuntimeError("Invalid color mapper for color image.")
    else:
        raise RuntimeError("Input image array must have either 2 or 3 dimensions.")

    # Get shape, dimensions
    n, m = im.shape[:2]
    if x_range is not None and y_range is not None:
        dw = x_range[1] - x_range[0]
        dh = y_range[1] - y_range[0]
    else:
        dw = m * interpixel_distance
        dh = n * interpixel_distance
        x_range = [0, dw]
        y_range = [0, dh]

    # Set up figure with appropriate dimensions
    if frame_width is None:
        frame_width = int(m / n * frame_height)
    if colorbar:
        toolbar_location = "above"
    else:
        toolbar_location = "right"
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=x_range,
        y_range=y_range,
        title=title,
        toolbar_location=toolbar_location,
        tools="pan,box_zoom,wheel_zoom,save,reset",
    )

    if no_ticks:
        p.xaxis.major_label_text_font_size = "0pt"
        p.yaxis.major_label_text_font_size = "0pt"
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.minor_tick_line_color = None
    else:
        if x_axis_label is None:
            p.xaxis.axis_label = length_units
        else:
            p.xaxis.axis_label = x_axis_label
        if y_axis_label is None:
            p.yaxis.axis_label = length_units
        else:
            p.yaxis.axis_label = y_axis_label

    # Display the image
    if im.ndim == 2:
        if flip:
            im = im[::-1, :]
        im_bokeh = p.image(
            image=[im],
            x=x_range[0],
            y=y_range[0],
            dw=dw,
            dh=dh,
            color_mapper=color_mapper,
        )
    else:
        im_bokeh = p.image_rgba(
            image=[rgb_to_rgba32(im, flip=flip)],
            x=x_range[0],
            y=y_range[0],
            dw=dw,
            dh=dh,
        )

    # Make a colorbar
    if colorbar:
        if im.ndim == 3:
            warnings.warn("No colorbar display for RGB images.")
        else:
            color_bar = bokeh.models.ColorBar(
                color_mapper=color_mapper,
                label_standoff=12,
                border_line_color=None,
                location=(0, 0),
            )
            p.add_layout(color_bar, "right")

    if display_clicks:
        div = bokeh.models.Div(width=200)
        layout = bokeh.layouts.row(p, div)
        p.js_on_event(bokeh.events.Tap, _display_clicks(div, attributes=["x", "y"]))
        if return_im:
            return layout, im_bokeh
        else:
            return layout

    if return_im:
        return p, im_bokeh
    return p


def _display_clicks(div, attributes=[], style="float:left;clear:left;font_size=0.5pt"):
    """Build a suitable CustomJS to display the current event
    in the div model."""
    return bokeh.models.CustomJS(
        args=dict(div=div),
        code="""
        var attrs = %s; var args = [];
        for (var i=0; i<attrs.length; i++ ) {
            args.push(Number(cb_obj[attrs[i]]).toFixed(4));
        }
        var line = "<span style=%r>[" + args.join(", ") + "], </span>\\n";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if ( lines.length > 35 ) { lines.shift(); }
        div.text = lines.join("\\n");
    """
        % (attributes, style),
    )
