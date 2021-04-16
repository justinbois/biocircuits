import numpy as np
import scipy.integrate

from .. import reg

import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting

import colorcet


def _ffl_rhs(beta, gamma, kappa, n_xy, n_xz, n_yz, ffl, logic):
    """Return a function with call signature fun(yz, x) that computes
    the right-hand side of the dynamical system for an FFL. Here,
    `yz` is a length two array containing concentrations of Y and Z.
    """
    if ffl[:2].lower() in ("c1", "c3", "i1", "i3"):
        fy = lambda x: reg.act_hill(x, n_xy)
    else:
        fy = lambda x: reg.rep_hill(x, n_xy)

    if ffl[:2].lower() in ("c1", "i4"):
        if logic.lower() == "and":
            fz = lambda x, y: reg.aa_and(x, y, n_xz, n_yz)
        else:
            fz = lambda x, y: reg.aa_or(x, y, n_xz, n_yz)
    elif ffl[:2].lower() in ("c4", "i1"):
        if logic.lower() == "and":
            fz = lambda x, y: reg.ar_and(x, y, n_xz, n_yz)
        else:
            fz = lambda x, y: reg.ar_or(x, y, n_xz, n_yz)
    elif ffl[:2].lower() in ("c2", "i3"):
        if logic.lower() == "and":
            fz = lambda x, y: reg.ar_and(y, x, n_yz, n_xz)
        else:
            fz = lambda x, y: reg.ar_or(y, x, n_yz, n_xz)
    else:
        if logic.lower() == "and":
            fz = lambda x, y: reg.rr_and(x, y, n_xz, n_yz)
        else:
            fz = lambda x, y: reg.rr_or(x, y, n_xz, n_yz)

    def rhs(yz, t, x):
        y, z = yz
        dy_dt = beta * fy(kappa * x) - y
        dz_dt = gamma * (fz(x, y) - z)

        return np.array([dy_dt, dz_dt])

    return rhs


def solve_ffl(beta, gamma, kappa, n_xy, n_xz, n_yz, ffl, logic, t, t_step_down, x_0):
    """Solve an FFL. The dynamics are given by
    `rhs`, the output of `ffl_rhs()`.
    """
    if t[0] != 0:
        raise RuntimeError("time must start at zero.")

    rhs = _ffl_rhs(beta, gamma, kappa, n_xy, n_xz, n_yz, ffl, logic)

    # Integrate if we do not step down
    if t[-1] < t_step_down:
        return scipy.integrate.odeint(rhs, np.zeros(2), t, args=(x_0,))

    # Integrate up to step down
    t_during_step = np.concatenate((t[t < t_step_down], (t_step_down,)))
    yz_during_step = scipy.integrate.odeint(
        rhs, np.zeros(2), t_during_step, args=(x_0,)
    )

    # Integrate after step
    t_after_step = np.concatenate(((t_step_down,), t[t > t_step_down]))
    yz_after_step = scipy.integrate.odeint(
        rhs, yz_during_step[-1, :], t_after_step, args=(0,)
    )

    # Concatenate solutions
    if t_step_down in t:
        return np.vstack((yz_during_step[:-1, :], yz_after_step))
    else:
        return np.vstack((yz_during_step[:-1, :], yz_after_step[1:, :]))


def plot_ffl(
    beta=1.0,
    gamma=1.0,
    kappa=1.0,
    n_xy=1.0,
    n_xz=1.0,
    n_yz=1.0,
    ffl="c1",
    logic="and",
    t=np.linspace(0, 20, 200),
    t_step_down=10.0,
    x_0=1.0,
    normalized=False,
    **kwargs,
):
    yz = solve_ffl(
        beta, gamma, kappa, n_xy, n_xz, n_yz, ffl, logic, t, t_step_down, x_0
    )
    y, z = yz.transpose()

    # Generate x-values
    if t[-1] > t_step_down:
        t_x = np.array([-t_step_down / 10, 0, 0, t_step_down, t_step_down, t[-1]])
        x = np.array([0, 0, x_0, x_0, 0, 0], dtype=float)
    else:
        t_x = np.array([-t[-1] / 10, 0, 0, t[-1]])
        x = np.array([0, 0, x_0, x_0], dtype=float)

    # Add left part of y and z-values
    t = np.concatenate(((t_x[0],), t))
    y = np.concatenate(((0,), y))
    z = np.concatenate(((0,), z))

    # Normalize if necessary
    if normalized:
        x /= x.max()
        y /= y.max()
        z /= z.max()

    # Set up figure
    frame_height = kwargs.pop("frame_height", 175)
    frame_width = kwargs.pop("frame_width", 550)
    x_axis_label = kwargs.pop("x_axis_label", "dimensionless time")
    y_axis_label = kwargs.pop(
        "y_axis_label", f"{'norm. ' if normalized else ''}dimensionless conc."
    )
    x_range = kwargs.pop("x_range", [t.min(), t.max()])
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        x_range=x_range,
    )

    # Column data sources
    cds = bokeh.models.ColumnDataSource(dict(t=t, y=y, z=z))
    cds_x = bokeh.models.ColumnDataSource(dict(t=t_x, x=x))

    # Populate glyphs
    colors = colorcet.b_glasbey_category10
    p.line(source=cds_x, x="t", y="x", line_width=2, color=colors[0], legend_label="x")
    p.line(source=cds, x="t", y="y", line_width=2, color=colors[1], legend_label="y")
    p.line(source=cds, x="t", y="z", line_width=2, color=colors[2], legend_label="z")

    # Allow vanishing lines by clicking legend
    p.legend.click_policy = "hide"

    return p, cds, cds_x


def _ffl_callback(
    p,
    cds,
    cds_x,
    beta,
    gamma,
    kappa,
    n_xy,
    n_xz,
    n_yz,
    ffl,
    logic,
    t_step_down,
    x_0,
    normalized,
):
    # Time points based on current axis limits
    t = np.linspace(0, p.x_range.end, 400)

    # Solve the dynamics
    yz = solve_ffl(
        beta, gamma, kappa, n_xy, n_xz, n_yz, ffl, logic, t, t_step_down, x_0
    )
    y, z = yz.transpose()

    # Generate x-values
    if t[-1] > t_step_down:
        t_x = np.array([-t_step_down / 10, 0, 0, t_step_down, t_step_down, t[-1]])
        x = np.array([0, 0, x_0, x_0, 0, 0], dtype=float)
    else:
        t_x = np.array([-t[-1] / 10, 0, 0, t[-1]])
        x = np.array([0, 0, x_0, x_0], dtype=float)

    # Add left part of y and z-values
    t = np.concatenate(((t_x[0],), t))
    y = np.concatenate(((0,), y))
    z = np.concatenate(((0,), z))

    # Normalize if necessary
    if normalized:
        x /= x.max()
        y /= y.max()
        z /= z.max()

    # Update ColumnDataSource
    cds.data = dict(t=t, y=y, z=z)
    cds_x.data = dict(t=t_x, x=x)


def _ffl_widgets():
    param_sliders_kwargs = dict(start=0.1, end=10, step=0.1, value=1, width=125)
    hill_coeff_kwags = dict(start=0.1, end=10, step=0.1, value=1, width=125)

    widgets = dict(
        beta_slider=bokeh.models.Slider(title="β", **param_sliders_kwargs),
        gamma_slider=bokeh.models.Slider(title="γ", **param_sliders_kwargs),
        kappa_slider=bokeh.models.Slider(title="κ", **param_sliders_kwargs),
        n_xy_slider=bokeh.models.Slider(title="nxy", **hill_coeff_kwags),
        n_xz_slider=bokeh.models.Slider(title="nxz", **hill_coeff_kwags),
        n_yz_slider=bokeh.models.Slider(title="nyz", **hill_coeff_kwags),
        ffl_selector=bokeh.models.Select(
            title="Circuit",
            options=[
                f"{x}-FFL" for x in ["C1", "C2", "C3", "C4", "I1", "I2", "I3", "I4"]
            ],
            value="C1-FFL",
            width=125,
        ),
        logic_selector=bokeh.models.RadioButtonGroup(
            name="Logic", labels=["AND", "OR"], active=0, width=125
        ),
        t_step_down_slider=bokeh.models.Slider(
            title="step down time", start=0.1, end=21, step=0.1, value=10, width=125
        ),
        x_0_slider=bokeh.models.Slider(
            title="x₀", start=0.1, end=10, step=0.1, value=1, width=125
        ),
        normalize_toggle=bokeh.models.Toggle(
            label="Normalize", active=False, width=125
        ),
    )

    return widgets


def ffl_app():
    """Create a Bokeh app for exploring the dynamics of feed-forward loops
    in response to a step input.

    Returns
    -------
    app : function
        The `app` function can be used to invoke a Bokeh app.

    Notes
    -----
    .. To serve the app from the command line so it has its own page
       in the browser, you can create a `.py`, say called
       `ffl_app.py`, with the following contents:

       ```
       import biocircuits.apps
       import bokeh.plotting

       app = biocircuits.apps.promiscuous_222_app()

       app(bokeh.plotting.curdoc())
       ```
       Then, from the command line, run:
       `bokeh serve --show ffl_app.py`
    .. To run the app from a Jupyter notebook, do the following in a
       code cell:
       ```
       import biocircuits.apps
       import bokeh.io

       bokeh.io.output_notebook()

       app = biocircuits.apps.ffl_app()
       bokeh.io.show(app, notebook_url='localhost:8888')

       ```
       You may need to change the `notebook_url` as necessary.
    """

    def app(doc):
        p, cds, cds_x = plot_ffl()
        widgets = _ffl_widgets()

        def _callback(attr, old, new):

            _ffl_callback(
                p,
                cds,
                cds_x,
                widgets["beta_slider"].value,
                widgets["gamma_slider"].value,
                widgets["kappa_slider"].value,
                widgets["n_xy_slider"].value,
                widgets["n_xz_slider"].value,
                widgets["n_yz_slider"].value,
                widgets["ffl_selector"].value,
                ("AND", "OR")[widgets["logic_selector"].active],
                widgets["t_step_down_slider"].value,
                widgets["x_0_slider"].value,
                widgets["normalize_toggle"].active,
            )

        for widget_name, widget in widgets.items():
            try:
                widget.on_change("value", _callback)
            except:
                widget.on_change("active", _callback)

        sliders = bokeh.layouts.row(
            bokeh.layouts.Spacer(width=30),
            bokeh.layouts.column(
                widgets["beta_slider"], widgets["gamma_slider"], widgets["kappa_slider"]
            ),
            bokeh.layouts.Spacer(width=10),
            bokeh.layouts.column(
                widgets["n_xy_slider"], widgets["n_xz_slider"], widgets["n_yz_slider"]
            ),
            bokeh.layouts.Spacer(width=10),
            bokeh.layouts.column(widgets["t_step_down_slider"], widgets["x_0_slider"]),
        )

        selectors = bokeh.layouts.column(
            widgets["ffl_selector"],
            widgets["logic_selector"],
            widgets["normalize_toggle"],
        )

        # Final layout
        layout = bokeh.layouts.column(
            p,
            bokeh.layouts.Spacer(width=20),
            bokeh.layouts.row(sliders, bokeh.layouts.Spacer(width=20), selectors),
        )

        doc.add_root(layout)

    return app
