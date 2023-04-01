function f(x, beta, n) {
    /* Derivative of function for root finding for fixed points */
    return Math.pow(x, n) - beta * Math.pow(x, n - 1) + 1.0;
}


function df(x, beta, n) {
    /* Function for root finding for fixed points */
    return n * Math.pow(x, n - 1) - (n - 1) * beta * Math.pow(x, n - 2);
}


function fixedPoints(beta, n, epsilon) {
    /* Find fixed points for dimensionless autorepression system. */
    let xMin = beta * (n - 1.0) / n;
    let fMin = f(xMin, beta, n)

    if (Math.abs(fMin) < epsilon) return [xMin];
    else if (fMin > 0.0) return [];
    else if (Math.abs(n - 1) < 1e-7) {
        return[beta - 1.0];
    }
    else if (n < 1) {
        return [newtonSolve(1e-6, f, df, [beta, n])];
    }
    else if (Math.abs(n - 2) < 1e-7) {
        let sqrt_discr = Math.sqrt(Math.pow(beta, 2) - 4.0);
        return [(beta - sqrt_discr) / 2.0, (beta + sqrt_discr) / 2.0];
    }
    else {
        let x1;
        let x2;
        if (n < 2) x1 = newtonSolve(1e-6, f, df, [beta, n]);
        else {
            let xInfl = beta * (n - 2.0) / n;
            x1 = newtonSolve(xInfl, f, df, [beta, n]);
        }
        x2 = newtonSolve(beta, f, df, [beta, n]);

        return [x1, x2];
    }
}


function fixedPointsDimensional(beta, gamma, k, n, tol, maxiters, epsilon) {
    /* Find fixed points for dimensional autorepression system. */
    let fps = fixedPoints(beta / gamma / k, n, epsilon);

    let x = [];
    let y = [];

    if (fps.length > 0) {
        for (let i = 0; i < fps.length; i++) {
            x.push(k * fps[i]);
            y.push(beta * Math.pow(fps[i], n) / (1 + Math.pow(fps[i], n)));
        }
    }
    return [x, y];
}


function callback() {
	// Extract data from source and sliders
	let x = cds.data['x'];
	let fp = cds.data['fp'];
	let fd = cds.data['fd'];
	let gamma = gamma_slider.value;
	let beta = beta_slider.value;
	let k = k_slider.value;
	let n = n_slider.value;

	// Update plot
	for (let i = 0; i < x.length; i++) {
	    fp[i] = beta * Math.pow(x[i] / k, n) / (1 + Math.pow(x[i] / k, n));
	    fd[i] = gamma * x[i];
	}

	// Compute fixed points
	let tol = 1e-8;
	let maxiters = 200;
	let epsilon = 1e-14;
	let [xFixedPoints, yFixedPoints] = fixedPointsDimensional(
	    beta, gamma, k, n, tol, maxiters, epsilon
	);

	// Update data sources
	if (xFixedPoints.length == 0) {
	    cds_fp_stable.data['x'] = [0];
	    cds_fp_stable.data['y'] = [0];
	    cds_fp_unstable.data['x'] = [];
	    cds_fp_unstable.data['y'] = [];
	}
	else if (xFixedPoints.length == 1) {
	    if (n - 1 < 1+1e-7) {
	        cds_fp_stable.data['x'] = xFixedPoints;
	        cds_fp_stable.data['y'] = yFixedPoints;
	        cds_fp_unstable.data['x'] = [0];
	        cds_fp_unstable.data['y'] = [0];
	    }
	    else {
	        cds_fp_stable.data['x'] = [0];
	        cds_fp_stable.data['y'] = [0];
	        cds_fp_unstable.data['x'] = xFixedPoints;
	        cds_fp_unstable.data['y'] = yFixedPoints;
	    }

	}
	else {
	    cds_fp_stable.data['x'] = [0, xFixedPoints[1]];
	    cds_fp_stable.data['y'] = [0, yFixedPoints[1]];
	    cds_fp_unstable.data['x'] = [xFixedPoints[0]];
	    cds_fp_unstable.data['y'] = [yFixedPoints[0]];
	}

	// Emit changes
	cds.change.emit();
	cds_fp_stable.change.emit();
	cds_fp_unstable.change.emit();
}
