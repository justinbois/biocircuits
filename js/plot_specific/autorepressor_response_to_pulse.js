function negAutoRHS(x, t, beta0, gamma, k, n, ks, ns, sFun, sArgs=[]) {
	let xScalar = x[0]

	let s = sFun(t, ...sArgs);

	// Correct for x being numerically negative
	let xCorr = (xScalar > 0.0) ? xScalar : 0.0;

	let beta = beta0 * Math.pow(s / ks, ns) / (1 + Math.pow(s / ks, ns));

	return [beta / (1 + Math.pow(xCorr / k, n)) - gamma * xCorr];
} 


function unregRHS(x, t, beta0, gamma, ks, ns, sFun, sArgs=[]) {
	let xScalar = x[0]
	let s = sFun(t, ...sArgs);

	let beta = beta0 * Math.pow(s / ks, ns) / (1 + Math.pow(s / ks, ns));

	return [beta - gamma * xScalar];
}


function sPulse(t, t0, tau) {
	return Math.exp(-4.0 * Math.pow((t - t0) / tau, 2));
}


function callback() {
	let xRangeMax = xRange.end;
	let dt = 0.01;
	let x0 = [0.0];
	let t0 = t0Slider.value;
	let tau = tauSlider.value;
	let s = cds.data['s'];

	let t = linspace(0.0, xRangeMax, cds.data['t'].length);
	let sArgs = [t0, tau];
	let args = [
		Math.pow(10, logBeta0Slider.value),
		Math.pow(10, logGammaSlider.value),
		Math.pow(10, logkSlider.value),
		nSlider.value,
		Math.pow(10, logksSlider.value),
		nsSlider.value,
		sPulse,
		sArgs
	];
	let argsUnreg = [args[0], args[1], args[4], args[5], args[6], args[7]];

	// Integrate ODES
	let xSolve = rkf45(negAutoRHS, x0, t, args, dt)[0];
	let xUnregSolve = rkf45(unregRHS, x0, t, argsUnreg)[0];

	// Pulse for plotting
	for (let i = 0; i < t.length; i++) {
		s[i] = sPulse(t[i], t0, tau);
	}

	// Normalize if necessary
	if (normalizeToggle.active) {
		let xMax = Math.max(...xSolve);
		if (xMax > 0.0) xSolve = svMult(1.0 / xMax, xSolve);

		let xUnregMax = Math.max(...xUnregSolve);
		if (xUnregMax > 0.0) xUnregSolve = svMult(1.0 / xUnregMax, xUnregSolve);

		let sMax = Math.max(...s);
		if (sMax > 0.0) s = svMult(1.0 / sMax, s);

		yaxis.axis_label = 'normalized concentration';
	}
	else yaxis.axis_label = 'concentration';

	// Toggle legend visibility
	legend.visible = (legendToggle.active) ? true : false;

	cds.data['t'] = t;
	cds.data['x'] = xSolve;
	cds.data['x_unreg'] = xUnregSolve;

	cds.change.emit();
}
