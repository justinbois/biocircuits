import warnings

import numpy as np
import scipy.integrate

import matplotlib.streamplot

import bokeh.application
import bokeh.application.handlers
import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting

import colorcet

from .jsfunctions import jsfuns


def _sin_plot():
    """Test to plot a sine wave"""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    p = bokeh.plotting.figure(frame_height=200, frame_width=400)
    cds = bokeh.models.ColumnDataSource(dict(x=x, y=y))
    p.line(source=cds, x="x", y="y", line_width=2)
    f_slider = bokeh.models.Slider(
        title="f", start=0.1, end=10, value=1, step=0.005, width=150
    )

    js_code = (
        jsfuns["sin"]
        + """
    let x = cds.data['x'];
    let y = cds.data['y'];
    let f = f_slider.value;

    for (let i = 0; i < x.length; i++) {
        y[i] = sin(f * x[i]);
    }

    cds.change.emit();
    """
    )

    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, f_slider=f_slider), code=js_code
    )
    f_slider.js_on_change("value", callback)

    layout = bokeh.layouts.column(
        bokeh.layouts.row(bokeh.models.Spacer(width=60), f_slider, width=150),
        bokeh.models.Spacer(height=20),
        p,
    )

    return layout


def gaussian_pulse():
    """Make a plot of a Gaussian pulse/
    """
    # t/s data for plotting
    t_0 = 4.0
    tau = 2.0
    t = np.linspace(0, 10, 200)
    s = np.exp(-4 * (t - t_0) ** 2 / tau ** 2)

    # Place the data in a ColumnDataSource
    cds = bokeh.models.ColumnDataSource(dict(t=t, s=s))

    # Build the plot
    p = bokeh.plotting.figure(
        frame_height=200,
        frame_width=400,
        x_axis_label="time",
        y_axis_label="input signal",
        x_range=[0, 10],
        y_range=[-0.02, 1.1],
    )
    p.line(source=cds, x="t", y="s", line_width=2)

    t0_slider = bokeh.models.Slider(
        title="t₀", start=0, end=10, step=0.01, value=4.0, width=150
    )
    tau_slider = bokeh.models.Slider(
        title="τ", start=0, end=10, step=0.01, value=2.0, width=150
    )

    # JavaScript callback
    js_code = jsfuns["gaussian_pulse"] + "callback()"
    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, t0_slider=t0_slider, tau_slider=tau_slider), code=js_code,
    )
    t0_slider.js_on_change("value", callback)
    tau_slider.js_on_change("value", callback)

    # Lay out and return
    return bokeh.layouts.row(
        p, bokeh.models.Spacer(width=30), bokeh.layouts.column(t0_slider, tau_slider)
    )


def autorepressor_response_to_pulse():
    """Make an interactive plot of the response of an autorepressive
    circuit's response to a Gaussian pulse of induction. Also overlay
    response of unregulated circuit and approximate pulse itself.
    """
    def neg_auto_rhs_s_fun(x, t, beta0, gamma, k, n, ks, ns, s_fun, s_args):
        """
        Right hand side for negative autoregulation function, with s variable.
        Returns dx/dt.

        s_fun is a function of the form s_fun(t, *s_args), so s_args is a tuple
        containing the arguments to pass to s_fun.
        """
        # Compute s
        s = s_fun(t, *s_args)

        # Correct for x possibly being numerically negative as odeint() adjusts step size
        x = np.maximum(0, x)

        # Plug in this value of s to the RHS of the negative autoregulation model
        return neg_auto_rhs(x, t, beta0, gamma, k, n, ks, ns, s)

    def unreg_rhs(x, t, beta0, gamma, ks, ns, s):
        """
        Right hand side for constitutive gene expression
        modulated to only be active in the presence of s.
        Returns dx/dt.
        """
        return beta0 * (s / ks) ** ns / (1 + (s / ks) ** ns) - gamma * x


    def unreg_rhs_s_fun(x, t, beta0, gamma, ks, ns, s_fun, s_args):
        """
        Right hand side for unregulated function, with s variable.
        Returns dx/dt.

        s_fun is a function of the form s_fun(t, *s_args), so s_args is a tuple
        containing the arguments to pass to s_fun.
        """
        # Compute s
        s = s_fun(t, *s_args)

        # Plug in this value of s to the RHS of the negative autoregulation model
        return unreg_rhs(x, t, beta0, gamma, ks, ns, s)


    def s_pulse(t, t_0, tau):
        """
        Returns s value for a pulse centered at t_0 with duration tau.
        """
        # Return 0 is tau is zero, otherwise Gaussian
        return 0 if tau == 0 else np.exp(-4 * (t - t_0) ** 2 / tau ** 2)


    # Set up initial parameters
    # Time points we want for the solution
    t = np.linspace(0, 10, 200)

    # Initial condition
    x0 = 0.0

    # Parameters
    beta0 = 100
    gamma = 1
    k = 0.5
    n = 1
    s = 100
    ns = 10
    ks = 0.1
    s_args = (4.0, 2.0)
    args = (beta0, gamma, k, n, ks, ns, s_pulse, s_args)
    args_unreg = (beta0, gamma, ks, ns, s_pulse, s_args)

    # Integrate ODE
    x = scipy.integrate.odeint(neg_auto_rhs_s_fun, x0, t, args=args)
    x = x.transpose()[0]
    x_unreg = scipy.integrate.odeint(unreg_rhs_s_fun, x0, t, args=args_unreg)
    x_unreg = x_unreg.transpose()[0]

    # also calculate the input
    s = s_pulse(t, *s_args)

    # Normalize time courses
    x /= x.max()
    x_unreg /= x_unreg.max()

    # set up the column data source
    cds = bokeh.models.ColumnDataSource(dict(t=t, x=x, s=s, x_unreg=x_unreg))

    # set up plot
    p = bokeh.plotting.figure(
        frame_width=375,
        frame_height=250,
        x_axis_label="time",
        y_axis_label="normalized concentration",
        x_range=[t.min(), t.max()],
    )

    # Populate glyphs
    p.line(source=cds, x="t", y="x", line_width=2, color=colors[1], legend_label="x neg. auto.")
    p.line(source=cds, x="t", y="x_unreg", line_width=2, color=colors[2], legend_label="x unreg.")
    p.line(source=cds, x="t", y="s", line_width=2, color=colors[0], legend_label="s")

    # Place the legend
    p.legend.location = "top_left"

    # Build the widgets
    log_beta0_slider = bokeh.models.Slider(
        title="log₁₀ β₀", start=-1, end=2, step=0.1, value=np.log10(beta0), width=150
    )
    log_gamma_slider = bokeh.models.Slider(
        title="log₁₀ γ", start=-1, end=2, step=0.1, value=np.log10(gamma), width=150
    )
    log_k_slider = bokeh.models.Slider(
        title="log₁₀ k", start=-1, end=2, step=0.1, value=np.log10(k), width=150
    )
    n_slider = bokeh.models.Slider(
        title="n", start=0.1, end=10, step=0.1, value=2, width=150
    )
    log_ks_slider = bokeh.models.Slider(
        title="log₁₀ kₛ", start=-2, end=2, step=0.1, value=np.log10(ks), width=150
    )
    ns_slider = bokeh.models.Slider(
        title="nₛ", start=0.1, end=10, step=0.1, value=10, width=150
    )
    t0_slider = bokeh.models.Slider(
        title="t₀", start=0.01, end=10, step=0.01, value=4.0, width=150
    )
    tau_slider = bokeh.models.Slider(
        title="τ", start=0.01, end=10, step=0.01, value=2.0, width=150
    )
    normalize_toggle = bokeh.models.Toggle(label='Normalize', active=True, width=50)
    legend_toggle = bokeh.models.Toggle(label='Legend', active=True, width=50)

def autoactivator_fixed_points():
    """Make an interactive plot of fixed points for a potentially
    bistable autoactivator circuit.
    """
    # Parameters for first plot
    beta = 10
    k = 3
    n = 5
    gamma = 1

    # Theroetical curves
    x = np.linspace(0, 20, 400)
    fp = beta * (x / k) ** n / (1 + (x / k) ** n)
    fd = gamma * x

    # Set up sliders
    params = [
        dict(
            name="γ", start=0.1, end=4, step=0.1, value=gamma, long_name="gamma_slider",
        ),
        dict(
            name="β", start=0.1, end=15, step=0.1, value=beta, long_name="beta_slider",
        ),
        dict(name="k", start=1, end=5, step=0.1, value=k, long_name="k_slider"),
        dict(name="n", start=0.1, end=10, step=0.1, value=n, long_name="n_slider"),
    ]
    sliders = [
        bokeh.models.Slider(
            start=param["start"],
            end=param["end"],
            value=param["value"],
            step=param["step"],
            title=param["name"],
            width=100,
        )
        for param in params
    ]

    # Build plot
    p = bokeh.plotting.figure(
        frame_height=200,
        frame_width=300,
        x_axis_label="x",
        y_axis_label="production or removal rate",
        y_range=[-1, 16],
        x_range=[-1, 16],
        toolbar_location="above",
    )

    # Column data source for curves
    cds = bokeh.models.ColumnDataSource(dict(x=x, fp=fp, fd=fd))
    p.line(source=cds, x="x", y="fp", line_width=2)
    p.line(source=cds, x="x", y="fd", line_width=2, color="orange")

    # Column data sources for stable and unstable fixed points.
    # Values for initial parameters hard-coded to save coding up fixed-
    # point finding in Python; already done in JS code
    cds_fp_stable = bokeh.models.ColumnDataSource(dict(x=[0, 9.97546], y=[0, 9.97546]))
    cds_fp_unstable = bokeh.models.ColumnDataSource(dict(x=[2.37605], y=[2.37605]))
    p.circle(source=cds_fp_stable, x="x", y="y", color="black", size=10)
    p.circle(
        source=cds_fp_unstable,
        x="x",
        y="y",
        line_color="black",
        line_width=2,
        fill_color="white",
        size=10,
    )

    # JavaScript callback, updates fixed points using Newton's method
    js_code = js_code = (
        jsfuns["rootfinding"] + jsfuns["autoactivator_fixed_points"] + "callback()"
    )

    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds, cds_fp_stable=cds_fp_stable, cds_fp_unstable=cds_fp_unstable
        ),
        code=js_code,
    )

    # Use the `js_on_change()` method to call the custom JavaScript code.
    for param, slider in zip(params, sliders):
        callback.args[param["long_name"]] = slider
        slider.js_on_change("value", callback)

    # Lay out and return
    return bokeh.layouts.row(
        p,
        bokeh.models.Spacer(width=30),
        bokeh.layouts.column([bokeh.models.Spacer(height=20)] + sliders),
    )


def toggle_nullclines():
    """Make an interactive plot of nullclines and fixed points of
    the Gardner-Collins synthetic toggle switch.
    """
    # Set up sliders
    params = [
        dict(
            name="βx", start=0.1, end=20, step=0.1, value=10, long_name="beta_x_slider",
        ),
        dict(
            name="βy", start=0.1, end=20, step=0.1, value=10, long_name="beta_y_slider",
        ),
        dict(name="n", start=1, end=10, step=0.1, value=4, long_name="n_slider"),
    ]
    sliders = [
        bokeh.models.Slider(
            start=param["start"],
            end=param["end"],
            value=param["value"],
            step=param["step"],
            title=param["name"],
            width=150,
        )
        for param in params
    ]

    # Build base plot with starting parameters
    beta = 10
    n = 4

    # Compute nullclines
    x_y = np.linspace(0, 20, 400)
    y_x = np.linspace(0, 20, 400)
    x_x = beta / (1 + y_x ** n)
    y_y = beta / (1 + x_y ** n)

    cds = bokeh.models.ColumnDataSource(data=dict(x_x=x_x, x_y=x_y, y_x=y_x, y_y=y_y))

    # Make the plot
    p = bokeh.plotting.figure(
        frame_height=250,
        frame_width=250,
        x_axis_label="x",
        y_axis_label="y",
        x_range=[-1, 20],
        y_range=[-1, 20],
    )
    p.line(x="x_x", y="y_x", source=cds, line_width=2, legend_label="x nullcline")
    p.line(
        x="x_y",
        y="y_y",
        source=cds,
        line_width=2,
        color="orange",
        legend_label="y nullcline",
    )
    cds_stable = bokeh.models.ColumnDataSource(
        dict(x=[0.0009999, 9.99999999999], y=[9.99999999999, 0.0009999])
    )
    cds_unstable = bokeh.models.ColumnDataSource(
        dict(x=[1.533012798623252], y=[1.533012798623252])
    )
    p.circle(source=cds_stable, x="x", y="y", color="black", size=10)
    p.circle(
        source=cds_unstable,
        x="x",
        y="y",
        line_color="black",
        fill_color="white",
        line_width=2,
        size=10,
    )

    # Callback (uses JavaScript)
    js_code = jsfuns["rootfinding"] + jsfuns["toggle_nullclines"] + "callback()"

    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, cdsStable=cds_stable, cdsUnstable=cds_unstable), code=js_code
    )

    # We use the `js_on_change()` method to call the custom JavaScript code.
    for param, slider in zip(params, sliders):
        callback.args[param["long_name"]] = slider
        slider.js_on_change("value", callback)

    # Return layout
    return bokeh.layouts.row(
        p,
        bokeh.models.Spacer(width=30),
        bokeh.layouts.column(bokeh.models.Spacer(height=40), *sliders),
    )


def repressilator():
    """Replaces the plot of the protein-only repressilator. Replaces
    Python code:

    def repressilator_rhs(x, t, beta, n):
        '''
        Returns 3-array of (dx_1/dt, dx_2/dt, dx_3/dt)
        '''
        x_1, x_2, x_3 = x

        return np.array(
            [
                beta / (1 + x_3 ** n) - x_1,
                beta / (1 + x_1 ** n) - x_2,
                beta / (1 + x_2 ** n) - x_3,
            ]
        )


    # Initial condiations
    x0 = np.array([1, 1, 1.2])

    # Number of points to use in plots
    n_points = 1000

    # Widgets for controlling parameters
    beta_slider = bokeh.models.Slider(title="β", start=0, end=100, step=0.1, value=10)
    n_slider = bokeh.models.Slider(title="n", start=1, end=5, step=0.1, value=3)
    t_max_slider = bokeh.models.Slider(title="t_max", start=1, end=100, step=1, value=40)

    # Solve for species concentrations
    def _solve_repressilator(beta, n, t_max):
        t = np.linspace(0, t_max, n_points)
        x = scipy.integrate.odeint(repressilator_rhs, x0, t, args=(beta, n))

        return t, x.transpose()


    # Obtain solution for plot
    t, x = _solve_repressilator(beta_slider.value, n_slider.value, t_max_slider.value)

    # Build the plot
    colors = colorcet.b_glasbey_category10[:3]

    p_rep = bokeh.plotting.figure(
        frame_width=550, frame_height=200, x_axis_label="t", x_range=[0, t_max_slider.value]
    )

    cds = bokeh.models.ColumnDataSource(data=dict(t=t, x1=x[0], x2=x[1], x3=x[2]))
    labels = dict(x1="x₁", x2="x₂", x3="x₃")
    for color, x_val in zip(colors, labels):
        p_rep.line(
            source=cds,
            x="t",
            y=x_val,
            color=color,
            legend_label=labels[x_val],
            line_width=2,
        )

    p_rep.legend.location = "top_left"


    # Set up plot
    p_phase = bokeh.plotting.figure(
        frame_width=200, frame_height=200, x_axis_label="x₁", y_axis_label="x₂",
    )

    p_phase.line(source=cds, x="x1", y="x2", line_width=2)


    if fully_interactive_plots:
        # Set up callbacks
        def _callback(attr, old, new):
            t, x = _solve_repressilator(
                beta_slider.value, n_slider.value, t_max_slider.value
            )
            cds.data = dict(t=t, x1=x[0], x2=x[1], x3=x[2])
            p_rep.x_range.end = t_max_slider.value

        beta_slider.on_change("value", _callback)
        n_slider.on_change("value", _callback)
        t_max_slider.on_change("value", _callback)

        # Build layout
        repressilator_layout = bokeh.layouts.column(
            p_rep,
            bokeh.layouts.Spacer(height=10),
            bokeh.layouts.row(
                p_phase,
                bokeh.layouts.Spacer(width=70),
                bokeh.layouts.column(beta_slider, n_slider, t_max_slider, width=150),
            ),
        )

        # Build the app
        def repressilator_app(doc):
            doc.add_root(repressilator_layout)

        bokeh.io.show(repressilator_app, notebook_url=notebook_url)
    else:
        beta_slider.disabled = True
        n_slider.disabled = True
        t_max_slider.disabled = True

        # Build layout
        repressilator_layout = bokeh.layouts.column(
            p_rep,
            bokeh.layouts.Spacer(height=10),
            bokeh.layouts.row(
                p_phase,
                bokeh.layouts.Spacer(width=70),
                bokeh.layouts.column(
                    bokeh.layouts.column(beta_slider, n_slider, t_max_slider, width=150),
                    bokeh.models.Div(
                        text='''
    <p>Sliders are inactive. To get active sliders, re-run notebook with
    <font style="font-family:monospace;">fully_interactive_plots = True</font>
    in the first code cell.</p>
            ''',
                        width=250,
                    ),
                ),
            ),
        )

    bokeh.io.show(repressilator_layout)

    """

    def repressilator_rhs(x, t, beta, n):
        """
        Returns 3-array of (dx_1/dt, dx_2/dt, dx_3/dt)
        """
        x_1, x_2, x_3 = x

        return np.array(
            [
                beta / (1 + x_3 ** n) - x_1,
                beta / (1 + x_1 ** n) - x_2,
                beta / (1 + x_2 ** n) - x_3,
            ]
        )

    # Initial condiations
    x0 = np.array([1, 1, 1.2])

    # Number of points to use in plots
    n_points = 1000

    # Widgets for controlling parameters
    beta_slider = bokeh.models.Slider(
        title="β", start=0.01, end=100, step=0.01, value=10.0
    )
    n_slider = bokeh.models.Slider(title="n", start=1, end=5, step=0.1, value=3)
    t_max_slider = bokeh.models.Slider(
        title="t_max", start=1, end=100, step=1, value=40
    )

    # Solve for species concentrations
    def _solve_repressilator(beta, n, t_max):
        t = np.linspace(0, t_max, n_points)
        x = scipy.integrate.odeint(repressilator_rhs, x0, t, args=(beta, n))

        return t, x.transpose()

    # Obtain solution for plot
    t, x = _solve_repressilator(beta_slider.value, n_slider.value, t_max_slider.value)

    # Build the plot
    colors = colorcet.b_glasbey_category10[:3]

    p_rep = bokeh.plotting.figure(
        frame_width=550,
        frame_height=200,
        x_axis_label="t",
        x_range=[0, t_max_slider.value],
    )

    cds = bokeh.models.ColumnDataSource(data=dict(t=t, x1=x[0], x2=x[1], x3=x[2]))
    labels = dict(x1="x₁", x2="x₂", x3="x₃")
    for color, x_val in zip(colors, labels):
        p_rep.line(
            source=cds,
            x="t",
            y=x_val,
            color=color,
            legend_label=labels[x_val],
            line_width=2,
        )

    p_rep.legend.location = "top_left"

    # Set up plot
    p_phase = bokeh.plotting.figure(
        frame_width=200, frame_height=200, x_axis_label="x₁", y_axis_label="x₂",
    )

    p_phase.line(source=cds, x="x1", y="x2", line_width=2)

    # Set up callbacks
    js_code = (
        jsfuns["reg"]
        + jsfuns["ode"]
        + jsfuns["circuits"]
        + """
var t = cds.data['t'];

var beta = beta_slider.value;
var n = n_slider.value;
var t_max = t_max_slider.value;

t = linspace(0.0, t_max, t.length);

var x = rkf45(repressilator, [1.0, 1.0, 1.2], t, [beta, n], t[1] - t[0], 1e-7, 1e-3, 100);

cds.data['t'] = t;
cds.data['x1'] = x[0];
cds.data['x2'] = x[1];
cds.data['x3'] = x[2];

x_range.end = t_max;

cds.change.emit();
"""
    )
    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            x_range=p_rep.x_range,
            beta_slider=beta_slider,
            n_slider=n_slider,
            t_max_slider=t_max_slider,
        ),
        code=js_code,
    )

    def _callback(attr, old, new):
        t, x = _solve_repressilator(
            beta_slider.value, n_slider.value, t_max_slider.value
        )
        cds.data = dict(t=t, x1=x[0], x2=x[1], x3=x[2])
        p_rep.x_range.end = t_max_slider.value

    beta_slider.js_on_change("value", callback)
    n_slider.js_on_change("value", callback)
    t_max_slider.js_on_change("value", callback)

    # Build layout
    layout = bokeh.layouts.column(
        p_rep,
        bokeh.layouts.Spacer(height=10),
        bokeh.layouts.row(
            p_phase,
            bokeh.layouts.Spacer(width=70),
            bokeh.layouts.column(beta_slider, n_slider, t_max_slider, width=150),
        ),
    )

    return layout


def turing_dispersion_relation():
    """Plot of dispersion relation for Turing patterns.

    Replaces Python code:

    def dispersion_relation(k_vals, d, mu):
        lam = np.empty_like(k_vals)
        for i, k in enumerate(k_vals):
            A = np.array([[1-d*k**2,          1],
                          [-2*mu,    -mu - k**2]])
            lam[i] = np.linalg.eigvals(A).real.max()

        return lam

    d_slider = pn.widgets.FloatSlider(
        name="d", start=0.01, end=1, value=0.05, step=0.01, width=150
    )

    mu_slider = pn.widgets.FloatSlider(
        name="μ", start=0.01, end=2, value=1.5, step=0.005, width=150
    )


    @pn.depends(d_slider.param.value, mu_slider.param.value)
    def plot_dispersion_relation(d, mu):
        k = np.linspace(0, 10, 200)
        lam_max_real_part = dispersion_relation(k, d, mu)

        p = bokeh.plotting.figure(
            frame_width=350,
            frame_height=200,
            x_axis_label="k",
            y_axis_label="Re[λ-max]",
            x_range=[0, 10],
        )

        p.line(k, lam_max_real_part, color="black", line_width=2)

        return p


    pn.Column(
        pn.Row(pn.Spacer(width=50), d_slider, mu_slider),
        pn.Spacer(height=20),
        plot_dispersion_relation,
    )

    """
    d_slider = bokeh.models.Slider(
        title="d", start=0.01, end=1, value=0.05, step=0.01, width=150
    )

    mu_slider = bokeh.models.Slider(
        title="μ", start=0.01, end=2, value=1.5, step=0.005, width=150
    )

    k = np.linspace(0, 10, 500)
    k2 = k ** 2
    mu = mu_slider.value
    d = d_slider.value
    b = mu + (1.0 + d) * k2 - 1.0
    c = (mu + k ** 2) * (d * k ** 2 - 1.0) + 2.0 * mu
    discriminant = b ** 2 - 4.0 * c

    lam = np.empty_like(k)
    inds = discriminant <= 0
    lam[inds] = -b[inds] / 2.0

    inds = discriminant > 0
    lam[inds] = (-b[inds] + np.sqrt(discriminant[inds])) / 2.0

    cds = bokeh.models.ColumnDataSource(dict(k=k, lam=lam))

    p = bokeh.plotting.figure(
        frame_width=350,
        frame_height=200,
        x_axis_label="k",
        y_axis_label="Re[λ-max]",
        x_range=[0, 10],
    )

    p.line(source=cds, x="k", y="lam", color="black", line_width=2)

    js_code = """
    function dispersion_relation(mu, d, k) {
        let k2 = k**2;
        let b = mu + (1.0 + d) * k2 - 1.0;
        let c = (mu + k**2) * (d * k**2 - 1.0) + 2.0 * mu
        let discriminant = b**2 - 4.0 * c;

        if (discriminant < 0) {
            return -b / 2.0;
        }
        else {
            return (-b + Math.sqrt(discriminant)) / 2.0;
        }
    }

    let mu = mu_slider.value;
    let d = d_slider.value;
    let k = cds.data['k'];
    let lam = cds.data['lam'];

    for (let i = 0; i < k.length; i++) {
        lam[i] = dispersion_relation(mu, d, k[i]);
    }

    cds.change.emit();
    """
    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, d_slider=d_slider, mu_slider=mu_slider), code=js_code
    )
    mu_slider.js_on_change("value", callback)
    d_slider.js_on_change("value", callback)

    layout = bokeh.layouts.column(
        bokeh.layouts.row(
            bokeh.models.Spacer(width=60), d_slider, mu_slider, width=400
        ),
        bokeh.models.Spacer(height=20),
        p,
    )

    return layout


def lotka_volterra():
    """Make a plot of the Lotka-Volterra system
    """
    """Test to plot Lotka-Volterra"""
    t = np.linspace(0.0, 20.0, 500)

    # Sliders
    alpha_slider = bokeh.models.Slider(
        title="α", start=0.1, end=10, value=1, step=0.005, width=150
    )
    beta_slider = bokeh.models.Slider(
        title="β", start=0.1, end=10, value=1, step=0.005, width=150
    )
    gamma_slider = bokeh.models.Slider(
        title="γ", start=0.1, end=10, value=1, step=0.005, width=150
    )
    delta_slider = bokeh.models.Slider(
        title="δ", start=0.1, end=10, value=1, step=0.005, width=150
    )

    def lotka_volterra_rhs(xy, t, alpha, beta, gamma, delta):
        # Unpack
        x, y = xy

        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y

        return np.array([dxdt, dydt])

    # Initial conditions
    x0 = np.array([1.0, 3.0])

    # Solve
    xy = scipy.integrate.odeint(
        lotka_volterra_rhs,
        x0,
        t,
        (alpha_slider.value, beta_slider.value, gamma_slider.value, delta_slider.value),
    )
    x, y = xy.transpose()

    # Set up plots
    p_phase = bokeh.plotting.figure(
        frame_width=200, frame_height=200, x_axis_label="x", y_axis_label="y",
    )
    p = bokeh.plotting.figure(frame_width=550, frame_height=200, x_axis_label="t",)

    # The data source
    cds = bokeh.models.ColumnDataSource(dict(t=t, x=x, y=y))
    p_phase.line(source=cds, x="x", y="y", line_width=2)
    p.line(source=cds, x="t", y="x", line_width=2)
    p.line(source=cds, x="t", y="y", color="tomato", line_width=2)

    js_code = (
        jsfuns["ode"]
        + jsfuns["circuits"]
        + """
var x = cds.data['x'];
var y = cds.data['y'];
var alpha = alpha_slider.value;
var beta = beta_slider.value;
var gamma = gamma_slider.value;
var delta = delta_slider.value;

var args = [alpha, beta, gamma, delta];
var xy = rkf45(lotkaVolterra, [1.0, 3.0], timePoints, args);
x = xy[0];
y = xy[1];

cds.data['x'] = x;
cds.data['y'] = y;

cds.change.emit();
"""
    )

    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            timePoints=t,
            alpha_slider=alpha_slider,
            beta_slider=beta_slider,
            gamma_slider=gamma_slider,
            delta_slider=delta_slider,
        ),
        code=js_code,
    )
    alpha_slider.js_on_change("value", callback)
    beta_slider.js_on_change("value", callback)
    gamma_slider.js_on_change("value", callback)
    delta_slider.js_on_change("value", callback)

    # Build layout
    layout = bokeh.layouts.column(
        p,
        bokeh.layouts.Spacer(height=10),
        bokeh.layouts.row(
            p_phase,
            bokeh.layouts.Spacer(width=70),
            bokeh.layouts.column(
                alpha_slider, beta_slider, gamma_slider, delta_slider, width=150
            ),
        ),
    )

    return layout
