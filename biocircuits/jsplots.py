import warnings

import numpy as np
import scipy.integrate
import scipy.special

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


def phosphorylation_signal_cascade(plot_inf=False):
    def eval_array(x, fun, args=()):
        """Evaluate function fun(x, *args) for each entry in array x,
        where fun(x, *args) expects scalar x and returns scalar."""
        return np.array([fun(x_val, *args) for x_val in x])

    def log(x):
        """Logarithm of nonnegative number with no divide by zero error"""
        return np.log(x) if x > 0.0 else -np.inf

    def f(s, n):
        """Convenience function for transfer function"""
        if np.isinf(n):
            res = f_infinite_n(s)
        elif np.isclose(s, 1):
            res = 1 / (1 + n)
        else:
            res = s ** n * (1 - s) / (1 - s ** (n + 1))

        return res

    def f_infinite_n(s):
        """f when n is infinite"""
        return 0.0 if s < 1 else (s - 1) / s

    def log_fprime(s, n):
        """Log of the derivative of f, calculated careful with logsumexp
        trick."""
        if np.isinf(n):
            res = log_fprime_infinite_n(s)
        elif np.isclose(s, 1):
            res = np.log(n) - np.log(1 + n) - np.log(2)
        else:
            res = (n - 1) * np.log(s) - 2 * np.log(np.abs(1 - s**(n+1)))
            if s < 1:
                res += np.log(n) +  np.log(1 - s)
                res += np.log(1 - s * (1 - s**n) / n / (1 -s))
            else:
                res += (n + 1) * np.log(s) + np.log(1 - 1 / s ** n)
                res += np.log(1 - n * (s - 1) / s / (s**n - 1))

        return res

    def log_fprime_infinite_n(s):
        if s == 1:
            res = np.log(0.5)
        elif s > 1:
            res = -2 * np.log(s)
        else:
            res = -np.inf

        return res


    def log_transfer_function(s, xtot, n):
        tf = transfer_function(s, xtot, n)

        if tf > 0:
            res = np.log(tf)
        elif np.isinf(n).any():
            res = -np.inf
        else:
            # Approximate calculation for small s
            n_prod = np.array([np.prod(n[j + 1 :]) for j in range(len(n))])
            res = np.prod(n) * np.log(s) + np.sum(n_prod * np.log(xtot))

        return res

    def transfer_function(s, xtot, n):
        res = xtot[0] * f(s, n[0])
        for xtot_val, n_val in zip(xtot[1:], n[1:]):
            res = xtot_val * f(res, n_val)

        return res

    def log_tf_deriv(s, xtot, n):
        """Can compute the log of the transfer function
        derivative because it is always ≥ 0
        """
        if np.isinf(n).any() and s < 1:
            res = -np.inf
        else:
            # Compute arguments of f_prime function
            args = [s]
            for n_val, xtot_val in zip(n[:-1], xtot[:-1]):
                args.append(xtot_val * f(args[-1], n_val))

            if (np.array(args) == 0.0).any():
                log_tf = log_transfer_function(s, xtot, n)
                if np.isinf(log_tf):
                    res = -np.inf
                else:
                    res = np.sum(np.log(n)) - np.log(s) + log_tf
            else:
                res = np.sum(np.log(xtot))
                for n_val, arg in zip(n, args):
                    res += log_fprime(arg, n_val)

        return res

    def log_sensitivity(s, xtot, n):
        """Sensistivity, including low-s limit."""
        if np.isinf(n).any() and s < 1:
            res = -np.inf
        else:
            log_tf = log_transfer_function(s, xtot, n)
            if log_tf < -300:
                res = np.sum(np.log(n))
            else:
                res = np.log(s) - log_tf + log_tf_deriv(s, xtot, n)

        return res

    # Fixed quantities for plots
    s = np.logspace(-4, 4, 400)
    colors = list(bokeh.palettes.Blues7[:4][::-1]) + ["black"]

    fig_kwargs = dict(
        frame_width=300,
        frame_height=175,
        x_axis_type="log",
        x_axis_label="s",
        x_range=[s.min(), s.max()],
    )
    titles = dict(
        tf="transfer function", gain="gain", deriv="derivative", sens="sensitivity"
    )
    y_axis_labels = titles
    plots = {
        feature: bokeh.plotting.figure(
            **fig_kwargs,
            y_axis_label=y_axis_labels[feature],
            title=titles[feature],
            visible=False,
        )
        for feature in ["tf", "gain", "deriv", "sens"]
    }

    log_plots = {
        feature: bokeh.plotting.figure(
            **fig_kwargs,
            y_axis_type="log",
            y_axis_label=y_axis_labels[feature],
            title=titles[feature],
        )
        for feature in ["tf", "gain", "deriv", "sens"]
    }
    for feature in log_plots:
        log_plots[feature].y_range.start = 1e-4

    # Link x-ranges
    for key, p in plots.items():
        if key != "tf":
            p.x_range = plots["tf"].x_range
        log_plots[key].x_range = plots["tf"].x_range

    # Control widgers
    radio_button_group = bokeh.models.RadioButtonGroup(
        labels=["log", "linear"],
        active=0,
        width=100,
    )
    cascade_depth_spinner = bokeh.models.Spinner(
        low=1, high=5, step=1, value=3,
        width=100,
        title="cascade depth",
    )
    xtot_sliders = {str(depth): bokeh.models.Slider(
        start=-2,
        end=2,
        value=depth - 1 if depth < 4 else 0,
        step=0.1,
        width=100,
        title=r"$$x_\mathrm{tot}^{(" + str(depth) + r")}$$",
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(2)"
        ),
        visible=depth <= cascade_depth_spinner.value,
    ) for depth in [1, 2, 3, 4, 5]}

    # Pull our values of sliders for making data for initial plot
    cascade_depth = int(cascade_depth_spinner.value)
    xtot = np.array([10 ** xtot_sliders[str(i)].value for i in range(1, cascade_depth_spinner.value+1)])

    # Build column data source
    log_s = np.log(s)
    log_tf_1=eval_array(
        s, log_transfer_function, (xtot, np.array([1] * cascade_depth))
    )
    log_tf_2=eval_array(
        s, log_transfer_function, (xtot, np.array([2] * cascade_depth))
    )
    log_tf_3=eval_array(
        s, log_transfer_function, (xtot, np.array([3] * cascade_depth))
    )
    log_tf_4=eval_array(
        s, log_transfer_function, (xtot, np.array([4] * cascade_depth))
    )
    log_tf_inf=eval_array(
        s, log_transfer_function, (xtot, np.array([np.inf] * cascade_depth))
    )
    log_tf_baseline=eval_array(s, log_transfer_function, (xtot[:1], np.array([1])))
    log_deriv_1=eval_array(s, log_tf_deriv, (xtot, np.array([1] * cascade_depth)))
    log_deriv_2=eval_array(s, log_tf_deriv, (xtot, np.array([2] * cascade_depth)))
    log_deriv_3=eval_array(s, log_tf_deriv, (xtot, np.array([3] * cascade_depth)))
    log_deriv_4=eval_array(s, log_tf_deriv, (xtot, np.array([4] * cascade_depth)))
    log_deriv_inf=eval_array(
        s, log_tf_deriv, (xtot, np.array([np.inf] * cascade_depth))
    )
    log_deriv_baseline=eval_array(s, log_tf_deriv, (xtot[:1], np.array([1])))
    log_sens_1=eval_array(s, log_sensitivity, (xtot, np.array([1] * cascade_depth)))
    log_sens_2=eval_array(s, log_sensitivity, (xtot, np.array([2] * cascade_depth)))
    log_sens_3=eval_array(s, log_sensitivity, (xtot, np.array([3] * cascade_depth)))
    log_sens_4=eval_array(s, log_sensitivity, (xtot, np.array([4] * cascade_depth)))
    log_sens_inf=eval_array(
        s, log_sensitivity, (xtot, np.array([np.inf] * cascade_depth))
    )
    log_sens_baseline=eval_array(s, log_sensitivity, (xtot[:1], np.array([1])))

    data = dict(
        s=s,
        tf_1=np.exp(log_tf_1),
        tf_2=np.exp(log_tf_2),
        tf_3=np.exp(log_tf_3),
        tf_4=np.exp(log_tf_4),
        tf_inf=np.exp(log_tf_inf),
        tf_baseline=np.exp(log_tf_baseline),
        deriv_1=np.exp(log_deriv_1),
        deriv_2=np.exp(log_deriv_2),
        deriv_3=np.exp(log_deriv_3),
        deriv_4=np.exp(log_deriv_4),
        deriv_inf=np.exp(log_deriv_inf),
        deriv_baseline=np.exp(log_deriv_baseline),
        gain_1=np.exp(log_tf_1 - log_s),
        gain_2=np.exp(log_tf_2 - log_s),
        gain_3=np.exp(log_tf_3 - log_s),
        gain_4=np.exp(log_tf_4 - log_s),
        gain_inf=np.exp(log_tf_inf - log_s),
        gain_baseline=np.exp(log_tf_baseline - log_s),
        sens_1=np.exp(log_sens_1),
        sens_2=np.exp(log_sens_2),
        sens_3=np.exp(log_sens_3),
        sens_4=np.exp(log_sens_4),
        sens_inf=np.exp(log_sens_inf),
        sens_baseline=np.exp(log_sens_baseline),
    )

    cds = bokeh.models.ColumnDataSource(data=data)

    # Plot baselines
    plots["tf"].line(source=cds, x="s", y="tf_baseline", color="gray", line_width=2)
    log_plots["tf"].line(source=cds, x="s", y="tf_baseline", color="gray", line_width=2)
    plots["deriv"].line(
        source=cds, x="s", y="deriv_baseline", color="gray", line_width=2
    )
    log_plots["deriv"].line(
        source=cds, x="s", y="deriv_baseline", color="gray", line_width=2
    )
    plots["gain"].line(source=cds, x="s", y="gain_baseline", color="gray", line_width=2)
    log_plots["gain"].line(
        source=cds, x="s", y="gain_baseline", color="gray", line_width=2
    )
    plots["sens"].line(source=cds, x="s", y="sens_baseline", color="gray", line_width=2)
    log_plots["sens"].line(
        source=cds, x="s", y="sens_baseline", color="gray", line_width=2
    )

    radio_button_group.js_on_change(
        "active",
        bokeh.models.CustomJS(
            args=dict(
                **{
                    "p_" + feature: plots[feature]
                    for feature in ["tf", "gain", "deriv", "sens"]
                },
                **{
                    "p_" + feature + "_log": log_plots[feature]
                    for feature in ["tf", "gain", "deriv", "sens"]
                },
            ),
            code="""
      if (p_tf_log.visible == true) {
        p_tf_log.visible = false;
        p_tf.visible = true;
        p_gain_log.visible = false;
        p_gain.visible = true;
        p_deriv_log.visible = false;
        p_deriv.visible = true;
        p_sens_log.visible = false;
        p_sens.visible = true;
      }
      else {
        p_tf_log.visible = true;
        p_tf.visible = false;
        p_gain_log.visible = true;
        p_gain.visible = false;
        p_deriv_log.visible = true;
        p_deriv.visible = false;
        p_sens_log.visible = true;
        p_sens.visible = false;
      }
    """,
        ),
    )

    n_str = ["1", "2", "3", "4", "inf"] if plot_inf else ["1", "2", "3", "4"]

    for feature in ["tf", "gain", "deriv", "sens"]:
       for color, n in zip(colors, n_str):
            kwargs = dict(
                source=cds, x="s", y=feature + "_" + n, color=color, line_width=2
            )
            # if feature == "tf":
            #     kwargs["legend_label"] = "n → ∞" if n == "inf" else f"n = {n}"
            plots[feature].line(**kwargs)
            log_plots[feature].line(**kwargs)
        # if feature == "tf":
        #     plots[feature].legend.location = "bottom_right"
        #     log_plots[feature].legend.location = "bottom_right"

    # JavaScript callback
    js_code = jsfuns["cascade_signal"] + jsfuns["utils"] + "callback()"

    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            cascadeDepthSpinner=cascade_depth_spinner,
            xTotSliders=xtot_sliders,
            plotInf=plot_inf,
        ),
        code=js_code,
    )

    # Link sliders
    for depth in xtot_sliders:
        xtot_sliders[depth].js_on_change("value", callback)
    cascade_depth_spinner.js_on_change("value", callback)

    # Build layout
    layout = bokeh.layouts.row(
        bokeh.layouts.column(
            bokeh.layouts.row(
                bokeh.layouts.Spacer(width=20),
                radio_button_group,
            ),
            bokeh.layouts.Spacer(height=15),
            bokeh.layouts.row(
                bokeh.layouts.column(plots["tf"], log_plots["tf"]),
                bokeh.layouts.column(plots["gain"], log_plots["gain"]),
            ),
            bokeh.layouts.row(
                bokeh.layouts.column(plots["deriv"], log_plots["deriv"]),
                bokeh.layouts.column(plots["sens"], log_plots["sens"]),
            ),
        ),
        bokeh.layouts.Spacer(width=50),
        bokeh.layouts.column(
            bokeh.layouts.Spacer(height=70),
            cascade_depth_spinner,
            bokeh.layouts.Spacer(height=20),
            *tuple([xtot_sliders[str(depth)] for depth in range(1, 6)]),
        ),
    )

    return layout


def michaelis_menten_approx():
    def michaelis_menten_rhs(c, t, kappa, zeta):
        cs, ces, cp = c
        return np.array(
            [
                (-(1 - ces) * cs + (1 - kappa) * ces) / kappa,
                ((1 - ces) * cs - ces) / kappa / zeta,
                ces,
            ]
        )

    def approx_michaelis_menten(c0, t, kappa, zeta):
        """Analytical solution to the Michaelis-Menten equation."""
        cs0, ces0, cp0 = c0
        cs = scipy.special.lambertw(cs0 * np.exp(cs0 - t)).real
        ces = cs / (1 + cs)
        cp = cs0 + cp0 - cs - zeta * (ces0 + ces)

        return cs, ces, cp

    kappa_slider = bokeh.models.Slider(
        title="κ", start=0.01, end=1, value=0.5, step=0.01, width=100
    )
    zeta_slider = bokeh.models.Slider(
        title="ζ",
        start=-2,
        end=2,
        value=-2,
        step=0.05,
        width=100,
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(2)"
        ),
    )
    cs0_slider = bokeh.models.Slider(
        title="init. substr. conc.",
        start=-1,
        end=1,
        value=0.0,
        step=0.01,
        width=100,
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(2)"
        ),
    )

    def solve_mm(kappa, zeta, cs0):
        # Initial condition
        c0 = np.array([cs0, 0.0, 0.0])

        # Time points
        t = np.linspace(0, 10, 400)

        # Solve the full system
        c = scipy.integrate.odeint(michaelis_menten_rhs, c0, t, args=(kappa, zeta))
        cs, ces, cp = c.transpose()

        # Solve the approximate system
        cs_approx, ces_approx, cp_approx = approx_michaelis_menten(c0, t, kappa, zeta)

        return t, cs, ces, cp, cs_approx, ces_approx, cp_approx

    # Get solution for initial glyphs
    t, cs, ces, cp, cs_approx, ces_approx, cp_approx = solve_mm(
        kappa_slider.value, 10 ** zeta_slider.value, 10 ** cs0_slider.value
    )

    # Set up ColumnDataSource for plot
    cds = bokeh.models.ColumnDataSource(
        dict(
            t=t,
            cs=cs,
            ces=ces,
            cp=cp,
            cs_approx=cs_approx,
            ces_approx=ces_approx,
            cp_approx=cp_approx,
        )
    )

    # Make the plot
    p = bokeh.plotting.figure(
        frame_width=500,
        frame_height=250,
        x_axis_label="dimensionless time",
        y_axis_label="dimensionless concentration",
        x_range=[0, 10],
        y_range=[-0.02, 1.02],
    )

    colors = colorcet.b_glasbey_category10

    # Populate glyphs
    p.line(
        source=cds,
        x="t",
        y="ces",
        line_width=2,
        color=colors[0],
        legend_label="ES",
    )
    p.line(
        source=cds,
        x="t",
        y="cs",
        line_width=2,
        color=colors[1],
        legend_label="S",
    )
    p.line(
        source=cds,
        x="t",
        y="cp",
        line_width=2,
        color=colors[2],
        legend_label="P",
    )
    p.line(
        source=cds,
        x="t",
        y="ces_approx",
        line_width=4,
        color=colors[0],
        alpha=0.3,
    )
    p.line(
        source=cds,
        x="t",
        y="cs_approx",
        line_width=4,
        color=colors[1],
        alpha=0.3,
    )
    p.line(
        source=cds,
        x="t",
        y="cp_approx",
        line_width=4,
        color=colors[2],
        alpha=0.3,
    )

    p.legend.location = "center_right"

    # JavaScript callback
    js_code = (
        jsfuns["michaelis_menten_approx"]
        + jsfuns["utils"]
        + jsfuns["linalg"]
        + jsfuns["ode"]
        + "callback()"
    )
    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            kappaSlider=kappa_slider,
            zetaSlider=zeta_slider,
            cs0Slider=cs0_slider,
            xRange=p.x_range,
            yRange=p.y_range,
        ),
        code=js_code,
    )

    # Link sliders
    kappa_slider.js_on_change("value", callback)
    zeta_slider.js_on_change("value", callback)
    cs0_slider.js_on_change("value", callback)

    # Also trigger if x_range changes
    p.x_range.js_on_change("end", callback)

    # Build layout
    layout = bokeh.layouts.column(
        bokeh.layouts.row(
            bokeh.models.Spacer(width=40),
            kappa_slider,
            bokeh.models.Spacer(width=10),
            zeta_slider,
            bokeh.models.Spacer(width=10),
            cs0_slider,
        ),
        bokeh.models.Spacer(width=10),
        p,
    )

    return layout


def gaussian_pulse():
    """Make a plot of a Gaussian pulse."""
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
        args=dict(cds=cds, t0_slider=t0_slider, tau_slider=tau_slider),
        code=js_code,
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

    def neg_auto_rhs(x, t, beta0, gamma, k, n, ks, ns, s):
        """
        Right hand side for negative autoregulation motif with s dependence.
        Return dx/dt.
        """
        # Compute dx/dt
        return (
            beta0 * (s / ks) ** ns / (1 + (s / ks) ** ns) / (1 + (x / k) ** n)
            - gamma * x
        )

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
    colors = colorcet.b_glasbey_category10

    # Time points we want for the solution
    t = np.linspace(0, 10, 200)

    # Initial condition
    x0 = 0.0

    # Parameters
    beta0 = 100.0
    gamma = 1.0
    k = 1.0
    n = 1.0
    s = 100.0
    ns = 10.0
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
        frame_width=400,
        frame_height=250,
        x_axis_label="time",
        y_axis_label="normalized concentration",
        x_range=[t.min(), t.max()],
    )

    # Populate glyphs
    p.line(
        source=cds,
        x="t",
        y="x",
        line_width=2,
        color=colors[1],
        legend_label="x neg. auto.",
    )
    p.line(
        source=cds,
        x="t",
        y="x_unreg",
        line_width=2,
        color=colors[2],
        legend_label="x unreg.",
    )
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
        title="n", start=0.1, end=10, step=0.1, value=n, width=150
    )
    log_ks_slider = bokeh.models.Slider(
        title="log₁₀ kₛ", start=-2, end=2, step=0.1, value=np.log10(ks), width=150
    )
    ns_slider = bokeh.models.Slider(
        title="nₛ", start=0.1, end=10, step=0.1, value=ns, width=150
    )
    t0_slider = bokeh.models.Slider(
        title="t₀", start=0.01, end=10, step=0.01, value=s_args[0], width=150
    )
    tau_slider = bokeh.models.Slider(
        title="τ", start=0.01, end=10, step=0.01, value=s_args[1], width=150
    )
    normalize_toggle = bokeh.models.Toggle(label="Normalize", active=True, width=50)
    legend_toggle = bokeh.models.Toggle(label="Legend", active=True, width=50)

    # JavaScript callback, updates fixed points using Newton's method
    js_code = (
        jsfuns["linalg"]
        + jsfuns["ode"]
        + jsfuns["utils"]
        + jsfuns["autorepressor_response_to_pulse"]
        + "callback()"
    )

    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            p=p,
            t0Slider=t0_slider,
            tauSlider=tau_slider,
            logBeta0Slider=log_beta0_slider,
            logGammaSlider=log_gamma_slider,
            logkSlider=log_k_slider,
            nSlider=n_slider,
            logksSlider=log_ks_slider,
            nsSlider=ns_slider,
            normalizeToggle=normalize_toggle,
            legendToggle=legend_toggle,
            xRange=p.x_range,
            yaxis=p.yaxis[0],
            legend=p.legend[0],
        ),
        code=js_code,
    )

    # Use the `js_on_change()` method to call the custom JavaScript code.
    for slider in [
        t0_slider,
        tau_slider,
        log_beta0_slider,
        log_gamma_slider,
        log_k_slider,
        n_slider,
        log_ks_slider,
        n_slider,
        log_ks_slider,
        ns_slider,
    ]:
        slider.js_on_change("value", callback)

    # Execute callback with changes in toggles
    normalize_toggle.js_on_change("active", callback)
    legend_toggle.js_on_change("active", callback)

    # Also trigger if x_range changes
    p.x_range.js_on_change("end", callback)

    # Lay out and return
    layout = bokeh.layouts.row(
        p,
        bokeh.layouts.Spacer(width=30),
        bokeh.layouts.column(
            log_beta0_slider,
            log_gamma_slider,
            log_k_slider,
            n_slider,
            legend_toggle,
        ),
        bokeh.layouts.column(
            log_ks_slider,
            ns_slider,
            t0_slider,
            tau_slider,
            normalize_toggle,
        ),
    )
    return layout


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
            name="γ",
            start=0.1,
            end=4,
            step=0.1,
            value=gamma,
            long_name="gamma_slider",
        ),
        dict(
            name="β",
            start=0.1,
            end=15,
            step=0.1,
            value=beta,
            long_name="beta_slider",
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
            name="βx",
            start=0.1,
            end=20,
            step=0.1,
            value=10,
            long_name="beta_x_slider",
        ),
        dict(
            name="βy",
            start=0.1,
            end=20,
            step=0.1,
            value=10,
            long_name="beta_y_slider",
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


def protein_repressilator():
    """Plot the dynamics of a protein-only repressilator circuit."""

    def protein_repressilator_rhs(x, t, beta, n):
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

    # Solve for species concentrations
    def _solve_repressilator(beta, n, t_max):
        t = np.linspace(0, t_max, n_points)
        x = scipy.integrate.odeint(protein_repressilator_rhs, x0, t, args=(beta, n))

        return t, x.transpose()

    # Obtain solution for plot
    t, x = _solve_repressilator(beta_slider.value, n_slider.value, 40.0)

    # Build the plot
    colors = colorcet.b_glasbey_category10[:3]

    p_rep = bokeh.plotting.figure(
        frame_width=550,
        frame_height=200,
        x_axis_label="t",
        x_range=[0, 40.0],
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
        frame_width=200,
        frame_height=200,
        x_axis_label="x₁",
        y_axis_label="x₂",
    )

    p_phase.line(source=cds, x="x1", y="x2", line_width=2)

    # Set up callbacks
    js_code = (
        jsfuns["reg"]
        + jsfuns["ode"]
        + jsfuns["circuits"]
        + jsfuns["utils"]
        + jsfuns["linalg"]
        + jsfuns["proteinRepressilator"]
        + "callback()"
    )
    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            xRange=p_rep.x_range,
            betaSlider=beta_slider,
            nSlider=n_slider,
        ),
        code=js_code,
    )

    beta_slider.js_on_change("value", callback)
    n_slider.js_on_change("value", callback)
    p_rep.x_range.js_on_change("end", callback)

    # Build layout
    layout = bokeh.layouts.column(
        p_rep,
        bokeh.layouts.Spacer(height=10),
        bokeh.layouts.row(
            p_phase,
            bokeh.layouts.Spacer(width=70),
            bokeh.layouts.column(beta_slider, n_slider, width=150),
        ),
    )

    return layout


def repressilator():
    """Plot the dynamics of a repressilator circuit."""

    # Sliders
    beta_slider = bokeh.models.Slider(
        title="β",
        start=0,
        end=4,
        step=0.1,
        value=1,
        width=125,
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(2)"
        ),
    )
    gamma_slider = bokeh.models.Slider(
        title="γ",
        start=-3,
        end=0,
        step=0.1,
        value=0,
        width=125,
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(3)"
        ),
    )
    rho_slider = bokeh.models.Slider(
        title="ρ",
        start=-6,
        end=0,
        step=0.1,
        value=-3,
        width=125,
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(6)"
        ),
    )
    n_slider = bokeh.models.Slider(
        title="n",
        start=1,
        end=5,
        step=0.1,
        value=3,
        width=125,
    )

    def repressilator_rhs(mx, t, beta, gamma, rho, n):
        """
        Returns 6-array of (dm_1/dt, dm_2/dt, dm_3/dt, dx_1/dt, dx_2/dt, dx_3/dt)
        """
        m_1, m_2, m_3, x_1, x_2, x_3 = mx
        return np.array(
            [
                beta * (rho + 1 / (1 + x_3 ** n)) - m_1,
                beta * (rho + 1 / (1 + x_1 ** n)) - m_2,
                beta * (rho + 1 / (1 + x_2 ** n)) - m_3,
                gamma * (m_1 - x_1),
                gamma * (m_2 - x_2),
                gamma * (m_3 - x_3),
            ]
        )

    # Initial condiations
    x0 = np.array([0, 0, 0, 1, 1.1, 1.2])

    # Number of points to use in plots
    n_points = 1000

    # Solve for species concentrations
    def _solve_repressilator(log_beta, log_gamma, log_rho, n, t_max):
        beta = 10 ** log_beta
        gamma = 10 ** log_gamma
        rho = 10 ** log_rho
        t = np.linspace(0, t_max, n_points)
        x = scipy.integrate.odeint(repressilator_rhs, x0, t, args=(beta, gamma, rho, n))
        m1, m2, m3, x1, x2, x3 = x.transpose()
        return t, m1, m2, m3, x1, x2, x3

    t, m1, m2, m3, x1, x2, x3 = _solve_repressilator(
        beta_slider.value,
        gamma_slider.value,
        rho_slider.value,
        n_slider.value,
        40.0,
    )

    cds = bokeh.models.ColumnDataSource(
        dict(t=t, m1=m1, m2=m2, m3=m3, x1=x1, x2=x2, x3=x3)
    )

    p = bokeh.plotting.figure(
        frame_width=500,
        frame_height=200,
        x_axis_label="t",
        x_range=[0, 40.0],
    )

    colors = bokeh.palettes.d3["Category20"][6]
    m1_line = p.line(source=cds, x="t", y="m1", line_width=2, color=colors[1])
    x1_line = p.line(source=cds, x="t", y="x1", line_width=2, color=colors[0])
    m2_line = p.line(source=cds, x="t", y="m2", line_width=2, color=colors[3])
    x2_line = p.line(source=cds, x="t", y="x2", line_width=2, color=colors[2])
    m3_line = p.line(source=cds, x="t", y="m3", line_width=2, color=colors[5])
    x3_line = p.line(source=cds, x="t", y="x3", line_width=2, color=colors[4])

    legend_items = [
        ("m₁", [m1_line]),
        ("x₁", [x1_line]),
        ("m₂", [m2_line]),
        ("x₂", [x2_line]),
        ("m₃", [m3_line]),
        ("x₃", [x3_line]),
    ]
    legend = bokeh.models.Legend(items=legend_items)
    legend.click_policy = "hide"

    p.add_layout(legend, "right")

    # Build the layout
    layout = bokeh.layouts.column(
        bokeh.layouts.row(
            beta_slider,
            gamma_slider,
            rho_slider,
            n_slider,
            width=575,
        ),
        bokeh.layouts.Spacer(height=10),
        p,
    )

    # Set up callbacks
    js_code = (
        jsfuns["reg"]
        + jsfuns["ode"]
        + jsfuns["circuits"]
        + jsfuns["utils"]
        + jsfuns["linalg"]
        + jsfuns["repressilator"]
        + "callback()"
    )
    callback = bokeh.models.CustomJS(
        args=dict(
            cds=cds,
            xRange=p.x_range,
            betaSlider=beta_slider,
            rhoSlider=rho_slider,
            gammaSlider=gamma_slider,
            nSlider=n_slider,
        ),
        code=js_code,
    )

    beta_slider.js_on_change("value", callback)
    gamma_slider.js_on_change("value", callback)
    rho_slider.js_on_change("value", callback)
    n_slider.js_on_change("value", callback)
    p.x_range.js_on_change("end", callback)

    return layout


def simple_binding_sensitivity():
    """Make a simple plot of sensitivity for binding of A and B."""

    def sensitivity(a_tot, b_tot, Kd):
        b = a_tot - b_tot - Kd
        discrim = b ** 2 + 4 * a_tot * Kd

        return a_tot * (1 - b / np.sqrt(discrim)) / (-b + np.sqrt(discrim))

    a_tot = np.logspace(-2, 2, 1001)
    b_tot = 1.0
    Kd_vals = np.logspace(-3, 1, 5)

    colors = bokeh.palettes.Blues9[3:8]

    b0_slider = bokeh.models.Slider(
        title="total B conc. (µM)",
        start=-1,
        end=1,
        value=np.log10(b_tot),
        step=0.01,
        width=200,
        format=bokeh.models.CustomJSTickFormatter(
            code="return Math.pow(10, tick).toFixed(2)"
        ),
    )

    p = bokeh.plotting.figure(
        frame_width=400,
        height=325,
        x_axis_label="total A conc. (µM)",
        y_axis_label="sensitivity",
        x_axis_type="log",
        y_axis_type="log",
        x_range=[a_tot.min(), a_tot.max()],
    )

    cds = bokeh.models.ColumnDataSource(
        {
            **{"a0": a_tot},
            **{f"s{i}": sensitivity(a_tot, b_tot, Kd) for i, Kd in enumerate(Kd_vals)},
        }
    )

    for i, (color, Kd) in enumerate(zip(colors, Kd_vals)):
        p.line(
            source=cds,
            x="a0",
            y=f"s{i}",
            line_width=2,
            color=color,
            legend_label=str(Kd) + " µM",
        )

    span = bokeh.models.Span(location=10 ** b0_slider.value, dimension="height")
    p.add_layout(span)

    p.legend.location = "bottom_right"
    p.legend.title = "Kd"

    # Set up callbacks
    js_code = jsfuns["simple_binding_sensitivity"] + "callback()"
    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, b0Slider=b0_slider, span=span),
        code=js_code,
    )

    b0_slider.js_on_change("value", callback)

    return bokeh.layouts.column(
        bokeh.layouts.row(bokeh.models.Spacer(width=153), b0_slider), p
    )


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
    """Make a plot of the Lotka-Volterra system"""
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
        frame_width=200,
        frame_height=200,
        x_axis_label="x",
        y_axis_label="y",
    )
    p = bokeh.plotting.figure(
        frame_width=550,
        frame_height=200,
        x_axis_label="t",
    )

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
