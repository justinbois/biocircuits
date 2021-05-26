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


function nanArray(n) {
	var x = [];
	for (var i = 0; i < n; i++) x.push(NaN);
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
	var m = arguments[0].length;
	var n = arguments.length;

	var result = [];
	for (var i = 0; i < m; i++) {
		var element = 0.0;
		for (var j = 0; j < n; j++) {
			element += arguments[j][i];
		}
		result.push(element);
	}

	return result;
}


function svMultAdd(scalars, vectors) {
	/*
	 * Add a set of vectors together, each multiplied by a scalar.
	 */

	var m = vectors[0].length;
	var n = scalars.length;

	if (vectors.length != n) {
		console.warn('svMultAdd: Difference number of scalars and vectors.')
		return null;
	}

	var result = [];
	for (var i = 0; i < m; i++) {
		var element = 0.0;
		for (var j = 0; j < n; j++) {
			element += scalars[j] * vectors[j][i];
		}
		result.push(element);
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


function rkf45(f, initialCondition, timePoints, args, dt, tol, sBounds, hMin, enforceNonnegative, maxDeadSteps) {
	// Set up return variables
	var tSol = [timePoints[0]];
	var t = timePoints[0];
	var iMax = timePoints.length;
	var y = [initialCondition];
	var y0 = initialCondition;
	var i = 1;
	var nDeadSteps = 0;
	var deadStep = false;

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

	if (enforceNonnegative === undefined) {
		enforceNonnegative = true;
	}

	if (maxDeadSteps === undefined) {
		maxDeadSteps = 10;
	}

	while (i < iMax && nDeadSteps < maxDeadSteps) {
		nDeadSteps = 0;
		while (t < timePoints[i] && nDeadSteps < maxDeadSteps) {
			[y0, t, h, deadStep] = rkf45Step(f, y0, t, args, h, tol, sBounds, hMin);
			nDeadSteps = deadStep ? nDeadSteps + 1 : 0;
			if (enforceNonnegative) {
				y0 = y0.map(function(x) {
					if (x < 0.0) return 0.0; else return x;
				})
			}
		}
		if (t > tSol[tSol.length - 1]) {
			y.push(y0);
			tSol.push(t);
		}
		i += 1;
	}

	if (nDeadSteps == maxDeadSteps) {
		return nanArray(iMax);
	}
	var yInterp = interpolateSolution(timePoints, tSol, transpose(y));

	return yInterp;
}


function rkf45Step(f, y, t, args, h, tol, sBounds, hMin) {
	var k1 = svMult(h , f(y, t, ...args));

	var y2 = svMultAdd([0.25, 1.0], [k1, y]);
    var k2 = svMult(h, f(y2, t + 0.25 * h, ...args));

    var y3 = svMultAdd([0.09375, 0.28125, 1.0], [k1, k2, y]);
    var k3 = svMult(h, f(y3, t + 0.375 * h, ...args));

    var y4 = svMultAdd(
    	[1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 1.0],
    	[k1, k2, k3, y]
    );
    var k4 = svMult(h, f(y4, t + 12.0 * h / 13.0, ...args));

    var y5 = svMultAdd(
    	[8341.0 / 4104.0, -32832.0 / 4104.0, 29440.0 / 4104.0, -845.0 / 4104.0, 1.0],
    	[k1, k2, k3, k4, y]
    );
    var k5 = svMult(h, f(y5, t + h, ...args));

    var y6 = svMultAdd(
    	[-6080.0 / 20520.0, 
    	 41040.0 / 20520.0, 
    	 -28352.0 / 20520.0, 
    	 9295.0 / 20520.0, 
    	 -5643.0 / 20520.0,
    	 1.0],
    	[k1, k2, k3, k4, k5, y]
    );
    var k6 = svMult(h, f(y6, t + h / 2.0, ...args));

	// Calculate error (note that k2's contribution to the error is zero)
	var errorVector = svMultAdd(
		[209.0 / 75240.0, 
		 -2252.8 / 75240.0, 
		 -2197.0 / 75240.0, 
		 1504.8 / 75240.0, 
		 2736.0 / 75240.0],
		[k1, k3, k4, k5, k6]);
	var error = Math.max(...absVector(errorVector));

    // Either don't take a step or use the RK4 step
    if (error < tol || h <= hMin){
    	var yNew = svMultAdd(
    		[2375.0 / 20520.0, 
    		 11264.0 / 20520.0, 
    		 10985.0 / 20520.0, 
    		 -4104.0 / 20520.0,
    		 1.0],
    		[k1, k3, k4, k5, y]);
        t += h;
        var deadStep = false;
    }
    else {
        var yNew = y;
        var deadStep = true;
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
    return [yNew, t, Math.max(s * h, hMin), deadStep];
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