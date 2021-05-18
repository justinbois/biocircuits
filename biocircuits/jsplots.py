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
