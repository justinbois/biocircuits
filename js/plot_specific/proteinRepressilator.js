function proteinRepressilator(x, t, beta, n) {
	// Unpack
	var [x1, x2, x3] = x;

	return [
		beta * rep_hill(x3, n) - x1,
		beta * rep_hill(x1, n) - x2,
		beta * rep_hill(x2, n) - x3
	];
}


function callback() {
	let xRangeMax = xRange.end;
	let dt = 0.01;
	let x0 = [1.0, 1.0, 1.2];
	let beta = betaSlider.value;
	let n = nSlider.value;

	let t = linspace(0.0, xRangeMax, cds.data['t'].length);
	let args = [beta, n];

	// Integrate ODES
	let xSolve = rkf45(proteinRepressilator, x0, t, args, t[1] - t[0], 1e-7, 1e-3, 100);

	cds.data['t'] = t;
	cds.data['x1'] = xSolve[0];
	cds.data['x2'] = xSolve[1];
	cds.data['x3'] = xSolve[2];

	cds.change.emit();
}
