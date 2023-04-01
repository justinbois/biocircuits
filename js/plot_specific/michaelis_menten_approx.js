function michaelisMentenRHS(c, t, kappa, zeta){
    let [cs, ces, cp] = c;

    return [
    	(-(1.0 - ces) * cs + (1.0 - kappa) * ces) / kappa,
        ((1.0 - ces) * cs - ces) / kappa / zeta,
        ces
    ];
}


function approxMichaelisMentenRHS(c, t, kappa, zeta) {
    let cs = c[0];
    return [-cs / (1.0 + cs)];
}


function callback() {
	let xRangeMax = xRange.end;
	let dt = 0.01;
	let cs0 = Math.pow(10.0, cs0Slider.value);
	let cp0 = 0.0; // No product to start
	let ces0 = 0.0;
	let c0 = [cs0, 0.0, 0.0];
	let c0Approx = [cs0];
	let kappa = kappaSlider.value;
	let zeta = Math.pow(10.0, zetaSlider.value);

	let t = linspace(0.0, xRangeMax, cds.data['t'].length);
	let args = [kappa, zeta];

	// Integrate ODES
	let cSolve = rkf45(michaelisMentenRHS, c0, t, args, dt);
	let csApprox = rkf45(approxMichaelisMentenRHS, c0Approx, t, args)[0];

	// Compute product and enzyme conc from substrate in approximate regime
    let cesApprox = [];
    let cpApprox = [];
    for (let i = 0; i < csApprox.length; i++) {
    	cesApprox[i] = csApprox[i] / (1.0 + csApprox[i]);
    	cpApprox[i] = cs0 + cp0 - csApprox[i] + zeta * (ces0 - cesApprox[i]);
    }

	// Update data
	cds.data['t'] = t;
	cds.data['cs'] = cSolve[0];
	cds.data['ces'] = cSolve[1];
	cds.data['cp'] = cSolve[2];
	cds.data['cs_approx'] = csApprox;
	cds.data['ces_approx'] = cesApprox;
	cds.data['cp_approx'] = cpApprox;

	// Update y-range
	yRange.start = -0.02 * cs0;
	yRange.end = 1.02 * cs0;

	cds.change.emit();
	yRange.change.emit();
}
