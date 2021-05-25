function linspace(start, stop, n) {
	var x = [];
	var currValue = start;
	var step = (stop - start) / (n - 1);
	for (var i = 0; i < n; i++) {
		x.push(currValue);
		currValue += step;
	}
	return x;
}


function zeros(n) {
	var x = [];
	for (var i = 0; i < n; i++) x.push(0.0);
	return x;
}


function transpose(A) {
	var m = A.length;
	var n = A[0].length;
	var AT = [];

	for (var j = 0; j < n; j++) {
		var ATj = [];
		for (var i = 0; i < m; i++) {
			ATj.push(A[i][j]);
		}
		AT.push(ATj);
	}

	return AT;
}


function svMult(a, v) {
	/*
	 * Multiply vector v by scalar a.
	 */

	return v.map(function(x) {
		return a * x;
	});
}


function svAdd(a, v) {
	/*
	 * Add a scalar a to every element of vector v.
	 */

	return v.map(function(x) {
		return a + x;
	});
}


function vectorVectorAdd(v1, v2) {
	if (v1.length !== v2.length) {
		console.warn('v1 and v2 are not the same length in vectorVectorAdd.')
		return null;
	}

	var result = [];
	var n = v1.length;

	for (var i = 0; i < n; i++) {
		result.push(v1[i] + v2[i]);
	}

	return result;
}


function vectorAdd() {
	if (v1.length !== v2.length) {
		console.warn('v1 and v2 are not the same length in vectorVectorAdd.')
		return null;
	}

	m = arguments[0].length;
	n = arguments.length;


	var result = [];
	for (var i = 0; i < m; i++) {
		var element = 0.0;
		for (var j = 0; j < n; j++) {
			element += arguments[j][i];
		}
		result.push(element)
	}

	return result;
}


function absVector(v) {
	var result = [];
	for (var i = 0; i < v.length; i++) {
		result[i] = Math.abs(v[i]);
	}

	return result;
}


function lotkaVolterra(xy, t, alpha, beta, gamma, delta) {
	// Unpack
	var [x, y] = xy;

	var dxdt = alpha * x - beta * x * y;
	var dydt = delta * x * y - gamma * y;

	return [dxdt, dydt];
}


function rkf45(f, initialCondition, timePoints, args, dt, tol, sBounds, hMin) {
	// Set up return variables
	var tSol = [timePoints[0]];
	var t = timePoints[0];
	var iMax = timePoints.length;
	var y = [initialCondition];
	var y0 = initialCondition;
	var i = 1;

	// Default parameters
	if (dt === undefined) {
		var h = timePoints[1] - timePoints[0];
	}
	else {
		var h = dt;
	}

	if (tol === undefined) {
		tol = 1e-7;
	}

	if (sBounds === undefined) {
		sBounds = [0.1, 10.0];
	}

	if (hMin === undefined) {
		hMin = 0.0;
	}

	while (i < iMax) {
		while (t < timePoints[i]) {
			[y0, t, h] = rkf45Step(f, y0, t, args, h, tol, sBounds, hMin);
		}
		if (t > tSol[tSol.length - 1]) {
			y.push(y0);
			tSol.push(t);
		}
		i += 1;
	}

	var yInterp = interpolateSolution(timePoints, tSol, transpose(y));

	return yInterp;
}


function rkf45Step(f, y, t, args, h, tol, sBounds, hMin) {
	var k1 = svMult(h , f(y, t, ...args));

    var y2 = vectorVectorAdd(svMult(0.25, k1), y);
    var k2 = svMult(h, f(y2, t + 0.25 * h, ...args));

    var kadd = vectorVectorAdd(svMult(3.0, k1), svMult(9.0, k2));
    kadd = svMult(1.0 / 32.0, kadd);
    var y3 = vectorVectorAdd(kadd, y);
    var k3 = svMult(h, f(y3, t + 0.375 * h, ...args));

    kadd = vectorVectorAdd(svMult(1932.0, k1), svMult(-7200.0, k2));
    kadd = vectorVectorAdd(kadd, svMult(7296.0, k3));
    kadd = svMult(1.0 / 2197.0, kadd);
    var y4 = vectorVectorAdd(kadd, y);
    var k4 = svMult(h, f(y4, t + 12.0 * h / 13.0, ...args));

    kadd = vectorVectorAdd(svMult(8341.0, k1), svMult(-32832.0, k2));
    kadd = vectorVectorAdd(kadd, svMult(29440.0, k3));
    kadd = vectorVectorAdd(kadd, svMult(-845.0, k4));
    kadd = svMult(1.0 / 4104.0, kadd);
    var y5 = vectorVectorAdd(kadd, y);
    var k5 = svMult(h, f(y5, t + h, ...args));

    kadd = vectorVectorAdd(svMult(-6080.0, k1), svMult(41040.0, k2));
    kadd = vectorVectorAdd(kadd, svMult(-28352.0, k3));
    kadd = vectorVectorAdd(kadd, svMult(9295.0, k4));
    kadd = vectorVectorAdd(kadd, svMult(-5643.0, k5));
    kadd = svMult(1.0 / 20520.0, kadd);
    var y6 = vectorVectorAdd(kadd, y);
    var k6 = svMult(h, f(y6, t + h / 2.0, ...args));

	// Calculate error
	var k1Err = svMult(209.0, k1);
	// k2Err is zero
	var k3Err = svMult(-2252.8, k3);
	var k4Err = svMult(-2197.0, k4);
	var k5Err = svMult(1504.8, k5);
	var k6Err = svMult(2736.0, k6);

	var errorVector = vectorVectorAdd(k1Err, k3Err);
	errorVector = vectorVectorAdd(errorVector, k4Err);
	errorVector = vectorVectorAdd(errorVector, k5Err);
	errorVector = vectorVectorAdd(errorVector, k6Err);
	errorVector = svMult(1.0 / 75240.0, errorVector);
	var error = Math.max(...absVector(errorVector));

    // Either don't take a step or use the RK4 step
    if (error < tol || h <= hMin){
    	// Calculate new step
		var y1Step = svMult(2375.0, k1);
		// y2Step is zero
		var y3Step = svMult(11264.0, k3);
		var y4Step = svMult(10985.0, k4);
		var y5Step = svMult(-4104.0, k5);

		var yStep = vectorVectorAdd(y1Step, y3Step);
		yStep = vectorVectorAdd(yStep, y4Step);
		yStep = vectorVectorAdd(yStep, y5Step);
		yStep = svMult(1.0 / 20520.0, yStep);

        var yNew = vectorVectorAdd(y, yStep);
        t += h;
    }
    else {
        var yNew = y;
    }

    // Compute scaling for new step size
    var s;
    if (error === 0.0) {
        s = sBounds[1];
    }
    else {
        s = Math.pow(tol * h / 2.0 / error, 0.25);
    }
    if (s < sBounds[0]) {
        s = sBounds[0];
    }
    else if (s > sBounds[1]){
        s = sBounds[1];
    }

    // Return new y-values, new time, and updated step size h
    return [yNew, t, Math.max(s * h, hMin)];
}


function interpolate1d(x, xs, ys) {
	var y2s = naturalSplineSecondDerivs(xs, ys);

	var yInterp = x.map(function(xVal) {
		return splineEvaluate(xVal, xs, ys, y2s);
	});

	return yInterp;
}


function interpolateSolution(timePoints, t, y) {
	// Interpolate each row of y
	var yInterp = y.map(function(yi) {
		return interpolate1d(timePoints, t, yi);
	});

	return yInterp;
}


function naturalSplineSecondDerivs(xs, ys) {
	/*
	 * Compute the second derivatives for a cubic spline data
	 * measured at positions xs, ys.
	 * 
	 * The second derivatives are then used to evaluate the spline.
	 */

	var n = xs.length;

	// Storage used in tridiagonal solve
	var u = zeros(n);

	// Return value
	var y2s = zeros(n);

	// Solve trigiadonal matrix by decomposition
	for (var i = 1; i < n-1; i++) {
		var fracInterval = (xs[i] - xs[i-1]) / (xs[i+1] - xs[i-1]);
		var p = fracInterval * y2s[i-1] + 2.0;
		y2s[i]=(fracInterval - 1.0) / p;
		u[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]) - (ys[i] - ys[i-1]) / (xs[i] - xs[i-1]);
		u[i] = (6.0 * u[i] / (xs[i+1] - xs[i-1]) - fracInterval * u[i-1]) / p;
	}

	// Tridiagonal solve back substitution
	for (var k = n-2; k >= 0; k--) {
		y2s[k] = y2s[k] * y2s[k+1] + u[k];
	}

	return y2s;
}


function splineEvaluate(x, xs, ys, y2s) {
	/*
	 * Evaluate a spline computed from points xs, ys, with second derivatives
	 * y2s, as compute by naturalSplineSecondDerivs().
	 *
	 * Assumes that x and xs are sorted.
	 */
	var n = xs.length;

	// Indices bracketing where x is
	var lowInd = 0;
	var highInd = n - 1;

	// Perform bisection search to find index of x
	while (highInd - lowInd > 1) {
		var i = (highInd + lowInd) >> 1;
		if (xs[i] > x) {
			highInd = i;
		}
		else {
			lowInd = i;
		}
	}
	var h = xs[highInd] - xs[lowInd];
	var a = (xs[highInd] - x) / h;
	var b = (x - xs[lowInd]) / h;

	var y = a * ys[lowInd] + b * ys[highInd]
	y += ((a**3 - a) * y2s[lowInd] + (b**3 - b) * y2s[highInd]) * h**2 / 6.0;

	return y
}



// rkf45(lotkaVolterra, [1.0, 3.0], linspace(0.0, 20.0, 200), [1.0, 2.0, 3.0, 4.0], 0.01, 1e-7, [0.1, 10.0], 0.0)