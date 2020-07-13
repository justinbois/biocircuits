import os

import numpy as np
import pandas as pd

import eqtk

import bokeh.models
import bokeh.layouts
import bokeh.palettes
import bokeh.plotting


def _make_rxns(nA, nB, nL):
    rxns = ""
    # Dimer rxns
    for j in range(nL):
        for i in range(nA):
            rxns += f"A_{i+1} + L_{j+1} <=> D_{i+1}_{j+1}\n"

    # Trimer rxns
    for j in range(nL):
        for i in range(nA):
            for k in range(nB):
                rxns += f"D_{i+1}_{j+1} + B_{k+1} <=> T_{i+1}_{j+1}_{k+1}\n"

    return rxns


def _make_N(nA, nB, nL):
    rxns = _make_rxns(nA, nB, nL)
    N = eqtk.parse_rxns(rxns)

    # Impose ordering of names
    names = [f"A_{i+1}" for i in range(nA)]
    names += [f"B_{i+1}" for i in range(nB)]
    names += [f"L_{i+1}" for i in range(nL)]
    names += [f"D_{i+1}_{j+1}" for j in range(nL) for i in range(nA)]
    names += [
        f"T_{i+1}_{j+1}_{k+1}" for j in range(nL) for i in range(nA) for k in range(nB)
    ]

    # As a Numpy array
    return N[names].to_numpy(copy=True, dtype=float)


def _readout(epsilon, c):
    return np.dot(epsilon, c[:, -len(epsilon) :].transpose())


def _random_epsilon(nA, nB, nL):
    return np.random.dirichlet(np.ones(nA * nB * nL))


def _random_K(nA, nB, nL):
    # Dimerization equilibrium constants
    K_dimer = np.concatenate([np.random.dirichlet(np.ones(nA)) for i in range(nL)])

    # Trimerization equilibrium constants
    K_trimer = np.random.dirichlet(np.ones(nA * nB * nL))

    return np.concatenate((K_dimer, K_trimer))


def _make_c0_grid(nA, nB, nL, n, low_ligand_conc=0.001, high_ligand_conc=1000.0):
    log_low = np.log10(low_ligand_conc)
    log_high = np.log10(high_ligand_conc)
    # Ligand concentrations
    cL0 = np.logspace(log_low, log_high, n)
    cL0 = np.meshgrid(*tuple([cL0] * nL))

    # Initialize c0 and fixed_c
    c0 = np.zeros((n ** nL, nA + nB + nL + nA * nL + nA * nB * nL))
    fixed_c = np.ones_like(c0) * np.nan

    # Add ligand concentrations
    for i in range(nL):
        c0[:, i + nA + nB] = cL0[i].flatten()
        fixed_c[:, i + nA + nB] = cL0[i].flatten()

    # Random concentrations of receptors
    for i in range(nA):
        c0[:, i] = 10 ** np.random.uniform(log_low, log_high)
    for i in range(nB):
        c0[:, i + nA] = 10 ** np.random.uniform(log_low, log_high)

    return c0, fixed_c


def _make_c0_rim(n, logA, logB, log_low, log_high):
    # Initialize c0 and fixed_c
    c0 = np.zeros((2 * n, 18))
    fixed_c = np.zeros_like(c0) * np.nan

    # Ligand concentrations
    cL10 = np.concatenate(
        [[0], np.logspace(log_low, log_high, n), [10 ** log_high] * (n - 1)]
    )
    cL20 = np.concatenate(
        [[10 ** log_high] * (n - 1), np.logspace(log_high, log_low, n), [0]]
    )
    c0[:, 4] = cL10
    c0[:, 5] = cL20
    fixed_c[:, 4] = cL10
    fixed_c[:, 5] = cL20

    # Cannot fix a concentration to zero, so change fixed_c=0 values
    fixed_c[np.where(fixed_c == 0)] = np.nan

    # Concentrations of receptors
    c0[:, 0] = 10 ** logA[0]
    c0[:, 1] = 10 ** logA[1]
    c0[:, 2] = 10 ** logB[0]
    c0[:, 3] = 10 ** logB[1]

    return c0, fixed_c


def _lic_rls(s):
    a = s[0]
    b = s[-1]
    c = np.max(s)
    d = np.min(s)

    # Ensure a is the low level.
    if a > b:
        a, b = b, a

    lic = d / a - b / c
    rls = a / b

    return lic, rls


def _to_label_str(x, pretty_powers_of_ten):
    pretty_powers = {
        0.0001: "10⁻⁴",
        0.001: "10⁻³",
        0.01: "10⁻²",
        0.1: "10⁻¹",
        1.0: "10⁰",
        10.0: "10¹",
        100.0: "10²",
        1000.0: "10³",
        10000.0: "10⁴",
    }
    if pretty_powers_of_ten and 0.0001 <= x <= 10000.0:
        if x % 10 == 0 or (1 / x) % 10 == 0 or x == 1:
            return pretty_powers[x]
        else:
            pow_ten = int(np.floor(np.log10(x)))
            coeff = x / 10 ** pow_ten
            return "{0:.2f}".format(coeff) + "×" + pretty_powers[10 ** pow_ten]
    else:
        return "{0:.2e}".format(x)


def _param_dict_to_array(param_dict):
    return np.concatenate(
        (param_dict["K"], param_dict["epsilon"], param_dict["logA"], param_dict["logB"])
    )


_imbalance_params = {
    "logA": np.log10(np.array([0.070003, 0.0072265])),
    "logB": np.log10(np.array([14.505, 0.0020123])),
    "K": (
        0.55065,
        0.44935,
        0.032076,
        0.96792,
        0.10169,
        0.28462,
        0.038524,
        0.0066946,
        0.020614,
        0.27227,
        0.14488,
        0.1307,
    ),
    "epsilon": (
        0.012357,
        0.18003,
        0.48492,
        0.04415,
        0.096726,
        0.086891,
        0.047076,
        0.047853,
    ),
}

_additive_params = {
    "logA": np.log10(np.array([0.14356, 0.0031554])),
    "logB": np.log10(np.array([0.029711, 0.089175])),
    "K": (
        0.73952,
        0.26048,
        0.33582,
        0.66418,
        0.31139,
        0.10826,
        0.086969,
        0.019978,
        0.008519,
        0.092182,
        0.33719,
        0.035517,
    ),
    "epsilon": (
        0.052266,
        0.092732,
        0.35798,
        0.1635,
        0.0915,
        0.16255,
        0.076068,
        0.0034033,
    ),
}

_ratiometric_params = {
    "logA": np.log10(np.array([14.758, 85.588])),
    "logB": np.log10(np.array([107.92, 0.28909])),
    "K": (
        0.0016677,
        0.99833,
        0.71197,
        0.28803,
        0.010589,
        0.22728,
        0.24735,
        0.0040583,
        0.13076,
        0.24912,
        0.10161,
        0.029231,
    ),
    "epsilon": (
        0.098864,
        0.021308,
        0.00091882,
        0.22745,
        0.039245,
        0.16528,
        0.38461,
        0.062319,
    ),
}

_balance_params = {
    "logA": np.log10(np.array([2.2772, 270.11])),
    "logB": np.log10(np.array([0.2441, 1.2624])),
    "K": (
        0.74172,
        0.25828,
        0.66746,
        0.33254,
        0.22697,
        0.030042,
        0.22169,
        0.001393,
        0.17624,
        0.11554,
        0.047332,
        0.18079,
    ),
    "epsilon": (
        0.095076,
        0.017184,
        0.49264,
        0.032259,
        0.16782,
        0.073906,
        0.021015,
        0.1001,
    ),
}


def promiscuous_222_app(
    n=13, low_ligand_conc=0.001, high_ligand_conc=1000.0, pretty_powers_of_ten=True,
):
    """Create a Bokeh app for exploring promiscuous ligand-receptor binding
    for a system with two types of receptor A, two types of receptor B,
    and two ligands.

    Parameters
    ----------
    n : int, default 13
        Number of different ligand concentrations to consider. Ligand
        concentrations vary logarithmically over `low_ligand_conc` to
        `high_ligand_conc`.
    low_ligand_conc: float, default 0.001
        Lowest ligand concentration to consider.
    high_ligand_conc: float, default 1000.0
        Highest ligand concentration to consider.
    pretty_powers_of_ten : bool, default True
        If `True`, then attempts are made to make axis labels look
        "pretty", like `10²` instead of `1.0e+02`. Not guaranteed to
        work for all choices of ligand concentration.

    Returns
    -------
    app : function
        The `app` function can be used to invoke a Bokeh app.

    Notes
    -----
    .. To serve the app from the command line so it has its own page
       in the browser, you can create a `.py`, say called
       `promiscuous_app.py`, with the following contents:

       ```
       import biocircuits.vignettes
       import bokeh.plotting

       app = biocircuits.vignettes.promiscuous_222_app()

       app(bokeh.plotting.curdoc())
       ```
       Then, from the command line, run:
       `bokeh serve --show promiscuous_app.py`
    .. To run the app from a Jupyter notebook, do the following in a
       code cell:
       ```
       import biocircuits.vignettes
       import bokeh.io
       import bokeh.plotting

       bokeh.io.output_notebook()

       app = biocircuits.vignettes.promiscuous_222_app()
       bokeh.io.show(app, notebook_url='localhost:8888')

       ```
       You may need to change the `notebook_url` as necessary.
    """
    log_low = np.log10(low_ligand_conc)
    log_high = np.log10(high_ligand_conc)

    def app(doc):
        param_names = [
            "K11",
            "K21",
            "K12",
            "K22",
            "K111",
            "K112",
            "K211",
            "K212",
            "K121",
            "K122",
            "K221",
            "K222",
            "eps111",
            "eps112",
            "eps211",
            "eps212",
            "eps121",
            "eps122",
            "eps221",
            "eps222",
            "log10 A1",
            "log10 A2",
            "log10 B1",
            "log10 B2",
        ]

        # Sliders
        receptor_kwargs = dict(
            start=log_low, end=log_high, step=0.1, value=0, width=100
        )
        K_dimer_kwargs = dict(start=0.01, end=1, step=0.01, value=0.25, width=100)
        K_trimer_kwargs = dict(start=0.01, end=1, step=0.01, value=0.125, width=100)
        eps_kwargs = dict(start=0, end=1, step=0.01, value=0.125, width=100)
        sliders = {}
        for name in param_names:
            if "log" in name:
                kwargs = receptor_kwargs
            elif "eps" in name:
                kwargs = eps_kwargs
            elif len(name) == 3:
                kwargs = K_dimer_kwargs
            else:
                kwargs = K_trimer_kwargs

            sliders[name] = bokeh.models.Slider(
                title=name.translate(str.maketrans("012", "₀₁₂")).replace("eps", "ε"),
                **kwargs,
            )

        # Preset parameters buttons
        random_params_button = bokeh.models.Button(
            label="random", button_type="success", width=100
        )
        additive_params_button = bokeh.models.Button(
            label="additive", button_type="primary", width=100
        )
        ratiometric_params_button = bokeh.models.Button(
            label="ratiometric", button_type="primary", width=100
        )
        imbalance_params_button = bokeh.models.Button(
            label="imbalance", button_type="primary", width=100
        )
        balance_params_button = bokeh.models.Button(
            label="balance", button_type="primary", width=100
        )

        # Generate random sequences
        n_random_selector = bokeh.models.Select(
            title="# of random param sets",
            value="10",
            options=["10", "30", "100", "300", "1000"],
            width=165,
        )
        gen_random_params_button = bokeh.models.Button(
            label="Generate random params", button_type="success", width=165
        )
        status = bokeh.models.Div(text="", width=150)

        # Save current parameters button
        save_params_button = bokeh.models.Button(
            label="save current params", button_type="warning", width=165
        )
        outfile_input = bokeh.models.TextInput(
            title="file name", value="_tmp.csv", width=165
        )

        # Load current parameters button
        load_params_button = bokeh.models.Button(
            label="load params", button_type="warning", width=165
        )
        infile_input = bokeh.models.TextInput(title="file name", value="", width=165)
        load_status = bokeh.models.Div(text="", width=165)

        # Initial concentrations
        c0, fixed_c = _make_c0_grid(2, 2, 2, n)

        # Indices for periphery
        inds = np.concatenate(
            (np.arange(n ** 2 - n, n ** 2), np.arange(n - 1, n ** 2 - n, n)[::-1])
        )

        # Stoichiometric matrix
        N = _make_N(2, 2, 2)

        def params_from_sliders():
            # Set concentrations
            c0[:, 0] = 10 ** sliders["log10 A1"].value
            c0[:, 1] = 10 ** sliders["log10 A2"].value
            c0[:, 2] = 10 ** sliders["log10 B1"].value
            c0[:, 3] = 10 ** sliders["log10 B2"].value

            # Pull values ouff sliders
            K = np.array([sliders[name].value for name in param_names[:12]])
            epsilon = np.array([sliders[name].value for name in param_names[12:20]])

            # Rescale K and epsilon as per paper
            K[:2] /= K[:2].astype(float).sum()
            K[2:4] /= K[2:4].astype(float).sum()
            K[4:] /= K[4:].astype(float).sum()
            epsilon /= epsilon.astype(float).sum()

            return c0, K, epsilon

        def solve():
            # Solve grid
            c0, K, epsilon = params_from_sliders()
            c = eqtk.fixed_value_solve(c0=c0, fixed_c=fixed_c, N=N, K=K)
            s = _readout(epsilon, c)

            # Initial concentrations for high/zero ligand
            c0_rim = np.vstack((c0[0, :], c0[0, :]))
            c0_rim[:, 4] = np.array([0, high_ligand_conc])
            c0_rim[:, 5] = np.array([high_ligand_conc, 0])
            fixed_c_rim = np.ones_like(c0_rim) * np.nan
            fixed_c_rim[0, 5] = high_ligand_conc
            fixed_c_rim[1, 4] = high_ligand_conc

            # Solve with one ligand concentration set to zero and one high
            c_extremes = eqtk.fixed_value_solve(
                c0=c0_rim, fixed_c=fixed_c_rim, N=N, K=K
            )
            s_extremes = _readout(epsilon, c_extremes)
            s_rim = np.concatenate(((s_extremes[0],), s[inds], (s_extremes[1],)))

            return s, s_rim

        # Convert x and y values to strings
        x_str = [_to_label_str(val, pretty_powers_of_ten) for val in c0[:, 4]]
        y_str = [_to_label_str(val, pretty_powers_of_ten) for val in c0[:, 5]]

        # Set the locations of the ticks
        x_range = [
            _to_label_str(val, pretty_powers_of_ten)
            for val in sorted(list(np.unique(c0[:, 4])))
        ]
        y_range = [
            _to_label_str(val, pretty_powers_of_ten)
            for val in sorted(list(np.unique(c0[:, 5])))
        ]

        # Set up figures
        p = bokeh.plotting.figure(
            x_range=x_range,
            y_range=y_range,
            frame_width=300,
            frame_height=300,
            x_axis_label="L₁",
            y_axis_label="L₂",
            tools="pan,box_zoom,wheel_zoom,reset,hover,save",
            tooltips=[("L1", "@x"), ("L2", "@y"), ("S", "@z")],
            toolbar_location="above",
            align="end",
            title="matrix",
        )

        p_rim = bokeh.plotting.figure(
            x_range=[-0.05 * 2 * n, 1.05 * 2 * n],
            y_range=[-0.25, 1.05],
            frame_width=300,
            frame_height=125,
            x_axis_label="coordinate along rim",
            y_axis_label="normalize signal",
            toolbar_location="above",
            tools="save",
            align="end",
            title="rim",
        )

        p_lic_rls = bokeh.plotting.figure(
            frame_width=300,
            frame_height=125,
            x_range=[-1.05, 1.05],
            y_range=[-0.05, 1.05],
            x_axis_label="ligand interference coefficient",
            y_axis_label="relative ligand strength",
            toolbar_location="above",
            tools="pan,box_zoom,wheel_zoom,reset,hover,tap,save",
            tooltips=[("LIC", "@lic"), ("RLS", "@rls")],
            align="end",
            title="LIC/RLS",
        )

        # Shading on rim plot
        p_rim.varea(
            [-0.05 * 2 * n, n, 1.05 * 2 * n],
            [-0.25, -0.25, -0.25],
            [0, 0, -0.25],
            fill_alpha=0.15,
            fill_color="#fc8d62",
        )
        p_rim.varea(
            [-0.05 * 2 * n, n, 1.05 * 2 * n],
            [-0.25, -0.25, -0.25],
            [-0.25, 0, 0],
            fill_alpha=0.15,
            fill_color="#8da0cb",
        )
        p_rim.line([-0.05 * 2 * n, 1.05 * 2 * n], [0, 0], color="black")
        L1_label = bokeh.models.Label(
            x=1.02 * 2 * n,
            y=-0.175,
            text="L₁",
            text_color="#8da0cb",
            text_align="right",
            text_font_size="8pt",
        )
        L2_label = bokeh.models.Label(
            x=-0.02 * n, y=-0.175, text="L₂", text_color="#fc8d62", text_font_size="8pt"
        )
        p_rim.add_layout(L1_label)
        p_rim.add_layout(L2_label)

        # Set up the data sources
        source = bokeh.models.ColumnDataSource(
            dict(x_str=[], y_str=[], z_val=[], x=[], y=[], z=[])
        )
        source_rim = bokeh.models.ColumnDataSource(dict(x=[], y=[]))
        source_rim_area_L1 = bokeh.models.ColumnDataSource(dict(x=[], y=[]))
        source_rim_area_L2 = bokeh.models.ColumnDataSource(dict(x=[], y=[]))

        source_lic_rls_curr = bokeh.models.ColumnDataSource(dict(lic=[], rls=[]))

        source_lic_rls_data = {name: [] for name in param_names}
        source_lic_rls_data["lic"] = []
        source_lic_rls_data["rls"] = []
        source_lic_rls_cum = bokeh.models.ColumnDataSource(source_lic_rls_data)

        # Color mapper
        mapper = bokeh.models.LinearColorMapper(
            palette=bokeh.palettes.Viridis256, low=0, high=1
        )

        # Put in the glyphs, coloring by z_val
        p.rect(
            x="x_str",
            y="y_str",
            width=1,
            height=1,
            source=source,
            fill_color={"field": "z_val", "transform": mapper},
            line_color=None,
        )

        # Glyphs for periphery
        p_rim.circle(x="x", y="y", source=source_rim)

        # LIC/RLS glyphs
        p_lic_rls.circle(
            x="lic", y="rls", source=source_lic_rls_cum, size=3, fill_alpha=0,
        )
        p_lic_rls.circle(
            x="lic",
            y="rls",
            source=source_lic_rls_curr,
            size=3,
            fill_alpha=0,
            color="tomato",
        )

        # Style
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "9px"
        p.axis.major_label_standoff = 0
        if n != 7:
            p.xaxis.major_label_orientation = np.pi / 3

        # Callbacks
        def update(attr, old, new):
            s, s_rim = solve()

            # Scale to get full range of colormapper
            z_val = (s - s.min()) / (s.max() - s.min())

            # Compute LIC and RLS
            lic, rls = _lic_rls(s_rim)

            # All measurements
            s_all = np.concatenate((s, s_rim))

            # Normalize s_rim
            s_rim = (s_rim - s_all.min()) / (s_all.max() - s_all.min())

            # Update sources
            source.data = dict(
                x_str=x_str, y_str=y_str, z_val=z_val, x=c0[:, 4], y=c0[:, 5], z=s,
            )
            source_rim.data = dict(x=np.arange(2 * n + 1), y=s_rim)

            data_dict = {name: [sliders[name].value] for name in param_names}
            data_dict["lic"] = [lic]
            data_dict["rls"] = [rls]
            source_lic_rls_cum.stream(data_dict)
            source_lic_rls_curr.data = dict(lic=[lic], rls=[rls])

        def random_parameters(event=None):
            K_dict = {name: [] for name in param_names[:12]}
            eps_dict = {name: [] for name in param_names[12:20]}
            receptor_dict = {
                "log10 A1": [],
                "log10 A2": [],
                "log10 B1": [],
                "log10 B2": [],
            }
            lic_rls_dict = {"lic": [], "rls": []}

            for i in range(int(n_random_selector.value)):
                K = _random_K(2, 2, 2)
                epsilon = _random_epsilon(2, 2, 2)
                logA = np.random.uniform(log_low, log_high, size=2)
                logB = np.random.uniform(log_low, log_high, size=2)
                c0_rim, fixed_c_rim = _make_c0_rim(n, logA, logB, log_low, log_high)

                # Solve along the rim
                c_rim = eqtk.fixed_value_solve(c0=c0_rim, fixed_c=fixed_c_rim, N=N, K=K)
                s_rim = _readout(epsilon, c_rim)

                # Compute LIC and RLS
                lic, rls = _lic_rls(s_rim)

                for i, name in enumerate(param_names[:12]):
                    K_dict[name].append(K[i])
                for i, name in enumerate(param_names[12:20]):
                    eps_dict[name].append(epsilon[i])
                receptor_dict["log10 A1"].append(logA[0])
                receptor_dict["log10 A2"].append(logA[1])
                receptor_dict["log10 B1"].append(logB[0])
                receptor_dict["log10 B2"].append(logB[1])
                lic_rls_dict["lic"].append(lic)
                lic_rls_dict["rls"].append(rls)

            data_dict = {**lic_rls_dict, **K_dict, **eps_dict, **receptor_dict}
            source_lic_rls_cum.stream(data_dict)

        def save_current_parameters(event=None):
            c0, K, epsilon = params_from_sliders()
            receptor_dict = {
                "A1": [c0[0, 0]],
                "A2": [c0[0, 1]],
                "B1": [c0[0, 2]],
                "B2": [c0[0, 3]],
            }
            K_dict = {name: [K[i]] for i, name in enumerate(param_names[:12])}
            eps_dict = {name: [epsilon[i]] for i, name in enumerate(param_names[12:20])}

            df = pd.DataFrame({**receptor_dict, **K_dict, **eps_dict})
            fname = outfile_input.value
            if os.path.isfile(fname):
                out_str = df.to_csv(index=False, header=False)
                with open(fname, "a") as f:
                    f.write(out_str)
            else:
                df.to_csv(fname, index=False)

        def load_parameters(event=None):
            load_status.text = "<center>Loading....</center>"
            try:
                df = pd.read_csv(infile_input.value)
                df["log10 A1"] = np.log10(df["A1"])
                df["log10 A2"] = np.log10(df["A2"])
                df["log10 B1"] = np.log10(df["B1"])
                df["log10 B2"] = np.log10(df["B2"])

                K_dict = {name: [] for name in param_names[:12]}
                eps_dict = {name: [] for name in param_names[12:20]}
                receptor_dict = {
                    "log10 A1": [],
                    "log10 A2": [],
                    "log10 B1": [],
                    "log10 B2": [],
                }
                lic_rls_dict = {"lic": [], "rls": []}

                for i, r in df.iterrows():
                    K = np.array([r[name] for name in param_names[:12]])
                    epsilon = np.array([r[name] for name in param_names[12:20]])
                    logA = np.array([r["log10 A1"], r["log10 A2"]])
                    logB = np.array([r["log10 B1"], r["log10 B2"]])

                    c0_rim, fixed_c_rim = _make_c0_rim(n, logA, logB, log_low, log_high)

                    # Solve along the rim
                    c_rim = eqtk.fixed_value_solve(
                        c0=c0_rim, fixed_c=fixed_c_rim, N=N, K=K
                    )
                    s_rim = _readout(epsilon, c_rim)

                    # Compute LIC and RLS
                    lic, rls = _lic_rls(s_rim)

                    for i, name in enumerate(param_names[:12]):
                        K_dict[name].append(K[i])
                    for i, name in enumerate(param_names[12:20]):
                        eps_dict[name].append(epsilon[i])
                    receptor_dict["log10 A1"].append(logA[0])
                    receptor_dict["log10 A2"].append(logA[1])
                    receptor_dict["log10 B1"].append(logB[0])
                    receptor_dict["log10 B2"].append(logB[1])
                    lic_rls_dict["lic"].append(lic)
                    lic_rls_dict["rls"].append(rls)

                data_dict = {**lic_rls_dict, **K_dict, **eps_dict, **receptor_dict}
                source_lic_rls_cum.stream(data_dict)

                load_status.text = "<center>Load successful.</center>"
            except:
                load_status.text = "<center>Load failed.</center>"

        def remove_on_changes():
            for param in param_names:
                sliders[param].remove_on_change("value_throttled", update)

        def add_on_changes():
            for param in param_names:
                sliders[param].on_change("value_throttled", update)

        def params_reset_random(event=None):
            K = _random_K(2, 2, 2)
            epsilon = _random_epsilon(2, 2, 2)
            logA = np.random.uniform(log_low, log_high, size=2)
            logB = np.random.uniform(log_low, log_high, size=2)

            reset_params(np.concatenate((K, epsilon, logA, logB)))

        def params_reset_additive(event=None):
            reset_params(_param_dict_to_array(_additive_params))

        def params_reset_ratiometric(event=None):
            reset_params(_param_dict_to_array(_ratiometric_params))

        def params_reset_imbalance(event=None):
            reset_params(_param_dict_to_array(_imbalance_params))

        def params_reset_balance(event=None):
            reset_params(_param_dict_to_array(_balance_params))

        def reset_params(params):
            for name, val in zip(param_names, params):
                sliders[name].value = val
            update(None, None, None)

        def params_reset_from_selection(attr, old, new):
            try:
                ind = new[0]
                for name in param_names:
                    sliders[name].value = source_lic_rls_cum.data[name][ind]

                update(None, None, None)
            except:
                pass

        # Link sliders and buttons
        add_on_changes()
        random_params_button.on_click(params_reset_random)
        additive_params_button.on_click(params_reset_additive)
        ratiometric_params_button.on_click(params_reset_ratiometric)
        imbalance_params_button.on_click(params_reset_imbalance)
        balance_params_button.on_click(params_reset_balance)
        gen_random_params_button.on_click(random_parameters)
        save_params_button.on_click(save_current_parameters)
        load_params_button.on_click(load_parameters)

        # Run update to populate plots
        update(None, None, None)

        source_lic_rls_cum.selected.on_change("indices", params_reset_from_selection)

        # Build layout
        K_dimer = [sliders[name] for name in param_names[:4]]
        receptors = [sliders[name] for name in param_names if "log10" in name]
        col1 = K_dimer + receptors
        col2 = [sliders[name] for name in param_names[4:12]]
        col3 = [sliders[name] for name in param_names[12:20]]

        slider_row = bokeh.layouts.row(
            bokeh.layouts.column(*col1),
            bokeh.layouts.Spacer(width=10),
            bokeh.layouts.column(*col2),
            bokeh.layouts.Spacer(width=10),
            bokeh.layouts.column(*col3),
        )

        row1 = bokeh.layouts.row(p, bokeh.layouts.Spacer(width=25), slider_row)

        buttons = bokeh.layouts.column(
            bokeh.models.Spacer(height=20),
            bokeh.models.Div(text="<b>Preset parameters</b>", width=200),
            bokeh.layouts.row(
                additive_params_button, ratiometric_params_button, random_params_button,
            ),
            bokeh.layouts.row(imbalance_params_button, balance_params_button,),
        )

        row2 = bokeh.layouts.row(
            bokeh.layouts.Spacer(width=15),
            p_rim,
            bokeh.layouts.Spacer(width=25),
            buttons,
        )

        random_params = bokeh.layouts.row(
            bokeh.layouts.Spacer(height=20),
            bokeh.layouts.column(
                bokeh.layouts.Spacer(height=18), gen_random_params_button,
            ),
            bokeh.layouts.Spacer(width=15),
            n_random_selector,
        )

        save_params = bokeh.layouts.row(
            bokeh.layouts.column(bokeh.layouts.Spacer(height=18), save_params_button),
            bokeh.layouts.Spacer(width=15),
            outfile_input,
        )

        load_params = bokeh.layouts.row(
            bokeh.layouts.column(
                bokeh.layouts.Spacer(height=18), load_params_button, load_status
            ),
            bokeh.layouts.Spacer(width=15),
            infile_input,
        )

        params_col = bokeh.layouts.column(
            bokeh.models.Spacer(height=15),
            random_params,
            bokeh.layouts.Spacer(height=15),
            save_params,
            bokeh.layouts.Spacer(height=15),
            load_params,
        )

        row3 = bokeh.layouts.row(
            bokeh.layouts.Spacer(width=15),
            p_lic_rls,
            bokeh.layouts.Spacer(width=25),
            params_col,
        )

        instructions = bokeh.models.Div(
            text="""
            <b><center>Instructions</center></b>
            <div style="border: 1px solid black;">
            <ul>
            <li>In generating the plots, the values of the parameters of the sliders are rescaled such that K₁₁ + K₁₂ = 1, K₂₁ + K₂₂ = 1, the sum of all K's with three indices equals one, and the sum of all ε's equals 1.</li>
            <br />
            <li>The matrix plot uses color coding to show the magnitude of the response for various ligand concentrations. All units are arbitrary. The lowest signal is colored purple, and the highest is colored yellow.</li>
            <br />
            <li>The rim plot shows the magnitude of the response going from the upper left to upper right to lower right of the matrix plot, shouldered on the beginning and end by L₁ = 0 and L₂ = 0, respectively. the magnitude of the response is normalized such that the minimum value on the matrix plots is zero and the maximum value on the matrix plot is one.</li>
            <br />
            <li>The LIC/RLS plot shows the relative ligand strength (RLS) plotted against the ligand interference coefficient (LIC) for every parameter set encountered since the dashboard was launched. The red point on the LIC/RLS plot corresponds to the current parameter values of the sliders. All previously encountered parameter values are shown in blue. Click on a blue point to set the parameter values corresponding to that point.</li>
            <br />
            <li>Adjust sliders to change parameter values. The response to the sliders is throttled, so plots will only respond when you release our mouse button.</li>
            <br />
            <li>You can load preset parameter values by clicking the blue buttons. Clicking the green "random" button will adjust the sliders and plots for a random set of parameters.</li>
            <br />
            <li>To expedite exploration of parameters, you can generate many random parameter sets and populate the LIC/RLS plot with the results by clicking the "generate random params" button. The values of the sliders and the matrix and rim plot will not change, but you can click on any of the blue dots added to the LIC/RLS plot to set the sliders and plots to those parameters. You can choose how many sets of parameters you which to generate from the "# of random param sets" pull-down menu. You may have to wait several seconds if you generate more than 30 sets at a time.</li>
            <br />
            <li>To store the current parameter values, click the "save current params" button. The parameter values will be saved to a CSV file given in the adjacent "file name" text box. If the file already exists, the current parameter values will be appended.</li>
            <br />
            <li>To load sets of parameters, click the "load params" button and the parameters in file specified in the adjacent "file name" text box will be loaded. The parameters are loaded and a corresponding glyph is added to the LIC/RLS plot.</li>
            </ul>
            </div>
        """,
            width=560,
        )

        row4 = bokeh.layouts.row(bokeh.layouts.Spacer(width=110), instructions)

        doc.add_root(
            bokeh.layouts.column(
                row1,
                bokeh.layouts.Spacer(height=30),
                row2,
                bokeh.layouts.Spacer(height=30),
                row3,
                bokeh.layouts.Spacer(height=30),
                row4,
            )
        )

    return app
