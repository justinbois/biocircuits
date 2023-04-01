function repressilator(x, t, beta, rho, gamma, n) {
	// Unpack
	let [m1, m2, m3, x1, x2, x3] = x;

	return [
		beta * (rho + rep_hill(x3, n)) - m1,
		beta * (rho + rep_hill(x1, n)) - m2,
		beta * (rho + rep_hill(x2, n)) - m3,
		gamma * (m1 - x1),
		gamma * (m2 - x2),
		gamma * (m3 - x3)
	];
}


function callback() {
	let xRangeMax = xRange.end;
	let dt = 0.01;
	let x0 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.2];
	let beta = Math.pow(10, betaSlider.value);
	let gamma = Math.pow(10, gammaSlider.value);
	let rho = Math.pow(10, rhoSlider.value);
	let n = nSlider.value;

	let t = linspace(0.0, xRangeMax, cds.data['t'].length);
	let args = [beta, rho, gamma, n];

	// Integrate ODES
	let xSolve = rkf45(repressilator, x0, t, args, t[1] - t[0], 1e-7, 1e-3, 100);

	cds.data['t'] = t;
	cds.data['m1'] = xSolve[0];
	cds.data['m2'] = xSolve[1];
	cds.data['m3'] = xSolve[2];
	cds.data['x1'] = xSolve[3];
	cds.data['x2'] = xSolve[4];
	cds.data['x3'] = xSolve[5];

	cds.change.emit();
}
