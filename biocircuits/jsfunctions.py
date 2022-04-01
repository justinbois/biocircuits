jsfuns = {
    "rootfinding": """
/** 
 * Find a root of a scalar function f(x) using Newton's method.
 * @param {float} x0 - guess for root location
 * @param {function} f - function we are funding the root of with call signature f(x, ...args)
 * @param {function} df - derivative of function we are funding the root of with call signature df(x, ...args)
 * @param {array} args - arguments to pass to f and df
 * @param {float} tol - tolerance for convergence
 * @param {int} maxIter - maximum of Newton steps to take
 * @param {float} epsilon - small number. Abort if derivative is smaller than this.
 */
function newtonSolve(x0, f, df, args=[], tol=1e-8, maxIter=200, epsilon=1e-14) {
	let x = Infinity;
	let solved = false;

    for (let i = 0; i < maxIter; i++) {
        let y = f(x0, ...args);
        let yprime = df(x0, ...args);

        if (Math.abs(yprime) < epsilon) {
        	break;
		}
 
        x = x0 - y / yprime;

        if (Math.abs(x - x0) <= tol) {
        	solved = true;
        	break;
        }

        x0 = x;
    }

    if (solved) return x;
    else return null;
}


/** 
 * Find a root of a scalar function f(x) using Brent's method.
 * @param {function} f - function we are funding the root of with call signature f(x, ...args)
 * @param {float} lower - lower bound for root
 * @param {float} upper - upper bound for root
 * @param {function} df - derivative of function we are funding the root of with call signature df(x, ...args)
 * @param {float} tol - tolerance for convergence
 * @param {int} maxIter - maximum of Newton steps to take
 */
function brentSolve(f, lower, upper, args=[], tol=1e-8, maxIter=1000) {
	let a = lower;
	let b = upper;
    let fa = f(a, ...args);
    let fb = f(b, ...args);

    // We may have already guessed the solution
    if (Math.abs(fa) < tol) return a;
    if (Math.abs(fb) < tol) return b;

    // Solution is not bracketed
    if (fa * fb >= 0) return null;

    // c is where we are closing in on the root
    let c = a;
    let fc = fa;

  	let iter = 0;
    while (iter++ < maxIter) {
    	let prevStep = b - a;

	    // Make sure a has the larger function value
	    if (Math.abs(fc) < Math.abs(fb)) {      
	    	[a, b, c] = [b, c, b];
	    	[fa, fb, fc] = [fb, fc, fb];
	    }

	    // Next step toward root
	    let newStep = (c - b) / 2.0;

	    // Adjusted tolerance
	    let tolAdj = 1e-15 * Math.abs(b) + tol / 2;

	    // Found a root!
    	if (Math.abs(newStep) <= tolAdj || fb === 0 ) {
      		return b;
    	}

	    // Try interpolation
	    if (Math.abs(prevStep) > tolAdj && Math.abs(fa) > Math.abs(fb)) {
	    	let p;
	    	let q;
	    	let t1;
	    	let t2;
	    	let cb = c - b;
	    	if (a === c) { // a and c coincide, so try linear interpolation
	    		t1 = fb / fa;
	    		p = cb * t1;
	    		q = 1.0 - t1;
	    	}
	    	else { // Use inverse quadratic interpolation
	    		q = fa / fc;
	    		t1 = fb / fc;
	    		t2 = fb / fa;
	    		p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0));
	    		q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0);
	    	}

	    	// Fix the signs on p and q
	    	if (p > 0) q = -q;
	    	else p = -p;

	    	// Accept the step if it's not too large and falls in interval
	    	if (p < (0.75 * cb * q - Math.abs(tolAdj * q) / 2.0) 
	    		&& p < Math.abs(prevStep * q / 2.0)) { 
		        newStep = p / q;
	      	}
	    }

	    // If we can't do interpolation, do bisection
	    // First make sure step is not smaller than the tolerance
        if (Math.abs(newStep) < tolAdj) {
	        newStep = (newStep > 0) ? tolAdj : -tolAdj;
        }
    
        // Swap with the previous approximation
        a = b;
        fa = fb;

        // Take the step
        b += newStep;
        fb = f(b, ...args);
    
    	// Adjust c so that the sign of f(c) is opposite f(b)
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
          c = a; 
          fc = fa;
        }
    }

    // If we did not converge, return null
    return null;
}


""",
    "circuits": """
function lotkaVolterra(xy, t, alpha, beta, gamma, delta) {
	// Unpack
	var [x, y] = xy;

	var dxdt = alpha * x - beta * x * y;
	var dydt = delta * x * y - gamma * y;

	return [dxdt, dydt];
}


function lotkaVolterraIMEX(alpha, beta, gamma, delta) {
	f = function(xy) {
		var [x, y] = xy;
		return [-beta * x * y, delta * x * y];
	}

	cfun = (x) => [0.0, 0.0];

	Afun = (x) => [alpha, -gamma];

	diagonalA = true;

	return {f: f, cfun: cfun, Afun: Afun, diagonalA: true};
}


function cascade(yz, t, beta, gamma, n_x, n_y, x) {
	// Unpack
	var [y, z] = yz;

	var dy_dt = beta * act_hill(x, n_x) - y;
	var dz_dy = gamma * (act_hill(y, n_y) - z)

	return [dy_dt, dz_dt];
}


function cascade(yz, t, beta, gamma, n_x, n_y, x_fun, x_args) {
	var x = x_fun(t, ...x_args);

	return cascade(yz, t, beta, gamma, n_x, n_y, x);
}


function repressilator(x, t, beta, n) {
	// Unpack
	var [x1, x2, x3] = x;

	return [
		beta * rep_hill(x3, n) - x1,
		beta * rep_hill(x1, n) - x2,
		beta * rep_hill(x2, n) - x3
	]
}


// module.exports = {
//   repressilator,
//   lotkaVolterra,
//   lotkaVolterraIMEX
// };
""",
    "ode": """
function ij(i, j, n) {
    /*
     * Lexicographic indexing of 2D array represented as 1D.
     */

    return i * n + j;
}

function twoDto1D(A) {
    /*
     * Convert a 2D matrix to a 1D representation with row-based (C)
     * lexicographic ordering.
     */

    var m = A.length;
    var n = A[0].length;

    var A1d = [];
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            A1d.push(A[i][j]);
        }
    }

    return A1d;
}

function LUPDecompose(A, eps) {
    /*
     * LUP decomposition.
     */

    var i, j, k, imax;
    var maxA, absA;
    var Arow;
    var p = [];
    var n = A.length;
    var LU = shallowCopyMatrix(A);

    // Permutation matrix
    for (i = 0; i <= n; i++) p.push(i);

    for (i = 0; i < n; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < n; k++) {
            absA = Math.abs(LU[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        // Failure; singular matrix
        if (maxA < eps) return [null, null];

        if (imax != i) {
            // Pivot
            j = p[i];
            p[i] = p[imax];
            p[imax] = j;

            // Pivot rows of A
            Arow = LU[i];
            LU[i] = LU[imax];
            LU[imax] = Arow;

            // Count pivots
            p[n]++;
        }

        for (j = i + 1; j < n; j++) {
            LU[j][i] /= LU[i][i];

            for (k = i + 1; k < n; k++) LU[j][k] -= LU[j][i] * LU[i][k];
        }
    }

    return [LU, p];
}

function LUPSolve(LU, p, b) {
    /*
     * Solve a linear system where LU and p are stored as the
     * output of LUPDecompose().
     */

    var n = b.length;
    var x = [];

    for (var i = 0; i < n; i++) {
        x.push(b[p[i]]);
        for (var k = 0; k < i; k++) x[i] -= LU[i][k] * x[k];
    }

    for (i = n - 1; i >= 0; i--) {
        for (k = i + 1; k < n; k++) x[i] -= LU[i][k] * x[k];

        x[i] /= LU[i][i];
    }

    return x;
}

function solve(A, b) {
    /*
     * Solve a linear system using LUP decomposition.
     *
     * Returns null if singular.
     */

    var eps = 1.0e-14;
    var LU, p;

    [LU, p] = LUPDecompose(A, eps);

    // Return null if singular
    if (LU === null) return null;

    return LUPSolve(LU, p, b);
}

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

function shallowCopyMatrix(A) {
    /*
     * Make a shallow copy of a matrix.
     */

    var Ac = [];
    var n = A.length;
    for (i = 0; i < n; i++) {
        Ac.push([...A[i]]);
    }

    return Ac;
}

function nanArray() {
	/*
	 * Return a NaN array of shape given by arguments.
	 */
	if (arguments.length == 1) {
	    var x = [];
    	for (var i = 0; i < arguments[0]; i++) x.push(NaN);
	}
	else if (arguments.length == 2) {
	    var x = [];
    	for (var i = 0; i < arguments[0]; i++) {
			var xRow = [];
    		for (var j = 0; j < arguments[1]; j++) xRow.push(NaN);
    		x.push(xRow);
    	}
	}
	else {
		throw 'Must only have one or two arguments to nanArray().'
	}

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

function dot(v1, v2) {
    /*
     * Compute dot product v1 . v2.
     */

    var n = v1.length;
    var result = 0.0;
    for (var i = 0; i < n; i++) result += v1[i] * v2[i];

    return result;
}

function norm(v) {
    /*
     * 2-norm of a vector
     */

    return Math.sqrt(dot(v, v));
}

function mvMult(A, v, diagonalA) {
    /*
     * Compute dot product A . v, where A is a matrix.
     * If diagonalA is true, then A must be a 1-D array.
     */

    if (diagonalA) return elementwiseVectorMult(A, v);
    else {
	    return A.map(function (Arow) {
	        return dot(Arow, v);
	    });
	}
}

function svMult(a, v) {
    /*
     * Multiply vector v by scalar a.
     */

    return v.map(function (x) {
        return a * x;
    });
}

function smMult(a, A) {
    /*
     * Multiply matrix A by scalar a.
     */

    return A.map(function (Arow) {
        return svMult(a, Arow);
    });
}

function svAdd(a, v) {
    /*
     * Add a scalar a to every element of vector v.
     */

    return v.map(function (x) {
        return a + x;
    });
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

function elementwiseVectorDivide(v1, v2) {
    /*
     * Compute v1 / v2 elementwise.
     */

    var result = [];
    n = v1.length;

    for (var i = 0; i < n; i++) {
        result.push(v1[i] / v2[i]);
    }

    return result;
}

function elementwiseVectorMult(v1, v2) {
    /*
     * Compute v1 * v2 elementwise.
     */

    var result = [];
    n = v1.length;

    for (var i = 0; i < n; i++) {
        result.push(v1[i] * v2[i]);
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
        console.warn("svMultAdd: Difference number of scalars and vectors.");
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

function rkf45(
    f,
    initialCondition,
    timePoints,
    args,
    dt,
    tol,
    relStepTol,
    maxDeadSteps,
    sBounds,
    hMin,
    enforceNonnegative
) {
    // Set up return variables
    var tSol = [timePoints[0]];
    var t = timePoints[0];
    var iMax = timePoints.length;
    var y = [initialCondition];
    var y0 = initialCondition;
    var i = 1;
    var nDeadSteps = 0;
    var deadStep = false;

    // DEBUG
    var nSteps = 0;
    // END EDEBUG

    // Default parameters
    if (dt === undefined) var h = timePoints[1] - timePoints[0];
    else var h = dt;

    if (tol === undefined) tol = 1e-7;
    if (relStepTol === undefined) relStepTol = 0.0;
    if (sBounds === undefined) sBounds = [0.1, 10.0];
    if (hMin === undefined) hMin = 0.0;
    if (enforceNonnegative === undefined) enforceNonnegative = true;
    if (maxDeadSteps === undefined) maxDeadSteps = 10;

    while (i < iMax && nDeadSteps < maxDeadSteps) {
        nDeadSteps = 0;
        while (t < timePoints[i] && nDeadSteps < maxDeadSteps) {
            [y0, t, h, deadStep] = rkf45Step(
                f,
                y0,
                t,
                args,
                h,
                tol,
                relStepTol,
                sBounds,
                hMin
            );
            nDeadSteps = deadStep ? nDeadSteps + 1 : 0;
            if (enforceNonnegative) {
                y0 = y0.map(function (x) {
                    if (x < 0.0) return 0.0;
                    else return x;
                });
            }
            // DEBUG
            nSteps += 1;
            // END DEBUG
        }
        if (t > tSol[tSol.length - 1]) {
            y.push(y0);
            tSol.push(t);
        }
        i += 1;
    }

    // DEBUG
    console.log(nSteps);
    // END DEBUG

    if (nDeadSteps == maxDeadSteps) {
    	var yInterp = nanArray(initialCondition.length, iMax);
    }
    else var yInterp = interpolateSolution(timePoints, tSol, transpose(y));

    return yInterp;
}

function rkf45Step(f, y, t, args, h, tol, relStepTol, sBounds, hMin) {
    var k1 = svMult(h, f(y, t, ...args));

    var y2 = svMultAdd([0.25, 1.0], [k1, y]);
    var k2 = svMult(h, f(y2, t + 0.25 * h, ...args));

    var y3 = svMultAdd([0.09375, 0.28125, 1.0], [k1, k2, y]);
    var k3 = svMult(h, f(y3, t + 0.375 * h, ...args));

    var y4 = svMultAdd(
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 1.0],
        [k1, k2, k3, y]
    );
    var k4 = svMult(h, f(y4, t + (12.0 * h) / 13.0, ...args));

    var y5 = svMultAdd(
        [
            8341.0 / 4104.0,
            -32832.0 / 4104.0,
            29440.0 / 4104.0,
            -845.0 / 4104.0,
            1.0,
        ],
        [k1, k2, k3, k4, y]
    );
    var k5 = svMult(h, f(y5, t + h, ...args));

    var y6 = svMultAdd(
        [
            -6080.0 / 20520.0,
            41040.0 / 20520.0,
            -28352.0 / 20520.0,
            9295.0 / 20520.0,
            -5643.0 / 20520.0,
            1.0,
        ],
        [k1, k2, k3, k4, k5, y]
    );
    var k6 = svMult(h, f(y6, t + h / 2.0, ...args));

    // Calculate new step
    var yNew = svMultAdd(
        [
            2375.0 / 20520.0,
            11264.0 / 20520.0,
            10985.0 / 20520.0,
            -4104.0 / 20520.0,
            1.0,
        ],
        [k1, k3, k4, k5, y]
    );

    // Relative difference between steps
	var relChangeStep = norm(vectorAdd(yNew, svMult(-1.0, y))) / norm(yNew);

    // Calculate error (note that k2's contribution to the error is zero)
    var errorVector = svMultAdd(
        [
            209.0 / 75240.0,
            -2252.8 / 75240.0,
            -2197.0 / 75240.0,
            1504.8 / 75240.0,
            2736.0 / 75240.0,
        ],
        [k1, k3, k4, k5, k6]
    );
    var error = Math.max(...absVector(errorVector));

    // Either don't take a step or use the RK4 step
    if (error < tol || relChangeStep < relStepTol || h <= hMin) {
        t += h;
        var deadStep = false;
    } else {
        yNew = y;
        var deadStep = true;
    }

    // Compute scaling for new step size
    var s;
    if (error === 0.0) {
        s = sBounds[1];
    } else {
        s = Math.pow((tol * h) / 2.0 / error, 0.25);
    }
    if (s < sBounds[0]) {
        s = sBounds[0];
    } else if (s > sBounds[1]) {
        s = sBounds[1];
    }

    // Return new y-values, new time, and updated step size h
    return [yNew, t, Math.max(s * h, hMin), deadStep];
}

function dydt(y, t, f, cfun, Afun, fArgs, cfunArgs, AfunArgs, diagonalA) {
    /*
     * Right hand side of ODEs for initializing IMEX method with RKF.
     */

    n = y.length;
    var rhs = zeros(n);

    var A = Afun(t, ...AfunArgs);
    var c = cfun(t, ...cfunArgs);

    // Linear part
    var nonConstantLinear = diagonalA
        ? elementwiseVectorMult(A, y)
        : mvMult(A, y, diagonalA);
    var linearPart = vectorAdd(nonConstantLinear, c);

    // Nonlinear part
    var nonlinearPart = f(y, t, ...fArgs);

    return vectorAdd(nonlinearPart, linearPart);
}

function cnab2Step(u, c, A, f1, f0, g1, omega, k, diagonalA) {
    /*
     * Take a CNAB2 step.
     *
     * - u is the current value of the solution.
     * - c is the constant term.
     * - A is the matrix for the linear function.
     * - f1 is the nonlinear function evaluated at the current value of y.
     * - f0 is the nonlinear function evaluated at the previous value of y.
     * - g1 is the linear function evaluated at the current value of y.
     * - omega is the ratio of the most recent step size to the one before that.
     * - k is the current step size.
     * - diagonalA is true if A is diagonal. This leads to a *much* faster time step.
     *   If diagonalA is true, then A is provided only as the diagonal.
     */

    var invk = 1.0 / k;
    var b = vectorAdd(
        svMult(0.5, c),
        svMult(invk, u),
        svMult(1.0 + omega / 2.0, f1),
        svMult(-omega / 2.0, f0),
        svMult(0.5, g1)
    );

    if (diagonalA) {
        var Aaug = svAdd(invk, svMult(-0.5, A));
        var result = elementwiseVectorDivide(b, Aaug);
    } else {
        var n = A.length;
        var Aaug = smMult(-0.5, A);
        for (i = 0; i < n; i++) {
            Aaug[i][i] += invk;
        }
        var result = solve(Aaug, b);
    }

    return result;
}

function vsimexAdjustStepSizePID(
    k,
    relChange,
    relChangeStep,
    tol,
    kP,
    kI,
    kD,
    kBounds,
    sBounds
) {
    /*
     * Adjust step size using a PID controller.
     */
    var mult =
        Math.pow(relChange[1] / relChangeStep, kP) *
        Math.pow(tol / relChangeStep, kI) *
        Math.pow(relChange[0] ** 2 / relChange[1] / relChangeStep, kD);
    if (mult > sBounds[1]) mult = sBounds[1];
    else if (mult < sBounds[0]) mult = sBounds[0];

    var newk = mult * k;

    if (newk > kBounds[1]) newk = kBounds[1];
    else if (newk < kBounds[0]) newk = kBounds[0];

    return newk;
}

function vsimexAdjustStepSizeRejectedStep(
    k,
    relChangeStep,
    tol,
    kBounds,
    sBounds
) {
    /*
     * Adjust step for rejected step
     */

    var mult = tol / relChangeStep;
    if (mult < sBounds[0]) mult = sBounds[0];

    var newk = mult * k;
    if (newk < kBounds[0]) newk = kBounds[0];

    return newk;
}

function vsimexAdjustStepSizeFailedSolve(k, failedSolveS) {
    /*
     * Adjust step for failed solve. Bringing step size down will
     * eventually make matrix for linear solve positive definite.
     */

    return k * failedSolveS;
}

function vsimex(
    f,
    cfun,
    Afun,
    initialCondition,
    timePoints,
    fArgs,
    cfunArgs,
    AfunArgs,
    diagonalA,
    k0,
    kBounds,
    tol,
    tolBuffer,
    kP,
    kI,
    kD,
    sBounds,
    failedSolveS,
    enforceNonnegative,
    maxDeadSteps
) {
    /*
     *
     */

    // Defaults
    if (k0 === undefined) k0 = 1.0e-5;
    if (kBounds === undefined) kBounds = [1.0e-6, 100.0];
    if (tol === undefined) tol = 0.001;
    if (tolBuffer === undefined) tolBuffer = 0.01;
    if (kP === undefined) kP = 0.075;
    if (kI === undefined) kI = 0.175;
    if (kD === undefined) kD = 0.01;
    if (sBounds === undefined) sBounds = [0.1, 10.0];
    if (failedSolveS === undefined) failedSolveS = 0.1;
    if (enforceNonnegative == undefined) enforceNonnegative = true;
    if (maxDeadSteps === undefined) maxDeadSteps = 10;

    // Do RKF to get the first few time points
    var rkf45TimePoints = [
        timePoints[0],
        timePoints[0] + k0,
        timePoints[0] + 2.0 * k0,
    ];

    var args = [f, cfun, Afun, fArgs, cfunArgs, AfunArgs, diagonalA];
    var yRKF = rkf45(
        dydt,
        initialCondition,
        rkf45TimePoints,
        args,
        k0 / 10.0,
        tol,
        sBounds,
        0.0,
        enforceNonnegative,
        maxDeadSteps
    );

    yRKF = transpose(yRKF);

    // Set up variables for running CNAB2 VSIMEX
    var tSol = [timePoints[0]];
    var iMax = timePoints.length;
    var y = [initialCondition];
    var k = 2.0 * k0;
    var newk;
    var t = rkf45TimePoints[2];
    var y0 = yRKF[2];
    var i = 1;
    var nDeadSteps = 0;
    var deadStep = false;
    var c = cfun(t, ...cfunArgs);
    var A = Afun(t, ...AfunArgs);
    var f0 = f(initialCondition, timePoints[0], ...fArgs);
    var f1 = f(y0, t, ...fArgs);
    var g1 = vectorAdd(c, mvMult(A, y0, diagonalA));
    var omega = 1.0;
    var yStep;
    var relChangeStep;
    var relTol = tol * (1.0 + tolBuffer);
    var relChange = [
        norm(vectorAdd(y0, svMult(-1.0, yRKF[1]))) / norm(y0),
        norm(vectorAdd(yRKF[1], svMult(-1.0, initialCondition))) /
            norm(yRKF[1]),
    ];

    // DEBUG
    var nSteps = 3;
    // END EDEBUG

    while (i < iMax && nDeadSteps < maxDeadSteps) {
        nDeadSteps = 0;
        while (t < timePoints[i] && nDeadSteps < maxDeadSteps) {
            // Take CNAB2 step
            yStep = cnab2Step(y0, c, A, f1, f0, g1, omega, k, diagonalA);

            // Reject the step if failed to solve
            if (yStep === null) {
                newk = vsimexAdjustStepSizeFailedSolve(k, failedSolveS);
                omega *= newk / k;
                k = newk;
                nDeadSteps += 1;
                console.log("null yStep");
            } else {
                // Relative change
                relChangeStep =
                    norm(vectorAdd(yStep, svMult(-1.0, y0))) / norm(yStep);

                // Take step if below tolerance
                if (relChangeStep <= relTol) {
                    f0 = f(y0, t, ...fArgs);
                    t += k;
                    y0 = yStep;
                    f1 = f(y0, t, ...fArgs);
                    c = cfun(t, ...cfunArgs);
                    A = Afun(t, ...AfunArgs);
                    g1 = vectorAdd(c, mvMult(A, y0, diagonalA));
                    newk = vsimexAdjustStepSizePID(
                        k,
                        relChange,
                        relChangeStep,
                        tol,
                        kP,
                        kI,
                        kD,
                        kBounds,
                        sBounds
                    );
                    relChange = [relChange[1], relChangeStep];
                    omega = newk / k;
                    k = newk;
                    nDeadSteps = 0;
                }
                // Reject the step is not within tolerance
                else {
                    newk = vsimexAdjustStepSizeRejectedStep(
                        k,
                        relChangeStep,
                        tol,
                        kBounds,
                        sBounds
                    );
                    omega *= newk / k;
                    k = newk;
                    nDeadSteps += 1;
                }
            }
            if (enforceNonnegative) {
                y0 = y0.map(function (x) {
                    if (x < 0.0) return 0.0;
                    else return x;
                });
            }

            // DEBUG
		    nSteps += 1;
		    // END EDEBUG
        }
        if (t > tSol[tSol.length - 1]) {
            y.push(y0);
            tSol.push(t);
        }
        i += 1;
    }

    // DEBUG
    console.log(nSteps);
    // END EDEBUG

    if (nDeadSteps == maxDeadSteps) {
        return nanArray(initialCondition, iMax);
    }
    var yInterp = interpolateSolution(timePoints, tSol, transpose(y));

    return yInterp;
}

function interpolate1d(x, xs, ys) {
    var y2s = naturalSplineSecondDerivs(xs, ys);

    var yInterp = x.map(function (xVal) {
        return splineEvaluate(xVal, xs, ys, y2s);
    });

    return yInterp;
}

function interpolateSolution(timePoints, t, y) {
    // Interpolate each row of y
    var yInterp = y.map(function (yi) {
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
    for (var i = 1; i < n - 1; i++) {
        var fracInterval = (xs[i] - xs[i - 1]) / (xs[i + 1] - xs[i - 1]);
        var p = fracInterval * y2s[i - 1] + 2.0;
        y2s[i] = (fracInterval - 1.0) / p;
        u[i] =
            (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) -
            (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]);
        u[i] =
            ((6.0 * u[i]) / (xs[i + 1] - xs[i - 1]) - fracInterval * u[i - 1]) /
            p;
    }

    // Tridiagonal solve back substitution
    for (var k = n - 2; k >= 0; k--) {
        y2s[k] = y2s[k] * y2s[k + 1] + u[k];
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
        } else {
            lowInd = i;
        }
    }
    var h = xs[highInd] - xs[lowInd];
    var a = (xs[highInd] - x) / h;
    var b = (x - xs[lowInd]) / h;

    var y = a * ys[lowInd] + b * ys[highInd];
    y +=
        (((a ** 3 - a) * y2s[lowInd] + (b ** 3 - b) * y2s[highInd]) * h ** 2) /
        6.0;

    return y;
}


// module.exports = {
//   vsimex,
//   rkf45,
//   zeros, 
//   linspace
// };

// vsimex(lotkaVolterra, [1.0, 3.0], linspace(0.0, 20.0, 200), [1.0, 2.0, 3.0, 4.0], 0.01, 1e-7, [0.1, 10.0], 0.0)
// var lv = lotkaVolterraIMEX(1.0, 2.0, 3.0, 4.0);
// var sol = vsimex(lv.f, lv.cfun, lv.Afun, [1.0, 3.0], linspace(0.0, 20.0, 200), [], [], [], lv.diagonalA)

""",
    "reg": """
function rep_hill(x, n) {
	return 1.0 / (1.0 + Math.pow(x, n));
}


function act_hill(x, n) {
	return 1.0 - 1.0 / (1.0 + Math.pow(x, n));
}


function aa_and(x, y, nx, ny) {
	var xnx = Math.pow(x, nx);
	var yny = Math.pow(y, ny);
	return xnx * yny / (1.0 + xnx) / (1.0 + yny);
}


function aa_or(x, y, nx, ny) {
	var denom = (1.0 + Math.pow(x, nx)) * (1.0 + Math.pow(y, ny));
	return (denom - 1.0) / denom;
}


function aa_or_single(x, y, nx, ny) {
	var num = Math.pow(x, nx) + Math.pow(y, ny);
	return num / (1.0 + num);
}


function rr_and(x, y, nx, ny) {
	return 1.0 / (1.0 + Math.pow(x, nx)) / (1.0 + Math.pow(y, ny));
}


function rr_and_single(x, y, nx, ny) {
	return 1.0 / (1.0 + Math.pow(x, nx) + Math.pow(y, ny));
}


function rr_or(x, y, nx, ny) {
	var xnx = Math.pow(x, nx);
	var yny = Math.pow(y, ny);

	return (1.0 + xnx + yny) / (1.0 + xnx) / (1.0 + yny);
}


function ar_and(x, y, nx, ny) {
	xnx = Math.pow(x, nx);
	return xnx / (1.0 + xnx) / (1.0 + Math.pow(y, ny));
}


function ar_or(x, y, nx, ny) {
	var nxn = Math.pow(x, nx);
	var yny = Math.pow(y, ny);

	return (1.0 + xnx * (1.0 + yny)) / (1.0 + xnx) / (1.0 + yny);
}


function ar_and_single(x, y, nx, ny) {
	xnx = Math.pow(x, nx);

	return xnx / (1.0 + xnx + Math.pow(y, ny));
}


function ar_or_single(x, y, nx, ny) {
	xnx = Math.pow(x, nx);

	return (1.0 + xnx) / (1.0 + xnx + Math.pow(y, ny));
}


function dActHill(x, n) {
	xn = Math.pow(x, n);

	return n * Math.Pow(x, n - 1.0) / Math.pow((1 + Math.pow(x, n)), 2);
}


function dRepHill(x, n) {
	xn = Math.pow(x, n);

	return -n * Math.Pow(x, n - 1.0) / Math.pow((1 + Math.pow(x, n)), 2);
}


// module.exports = {
//   rep_hill,
//   act_hill
// };
""",
    "autorepressor_response_to_pulse": """
function ode_rhs....
CONTINUE HERE.
""",
    "gaussian_pulse": """
function sPulse(t, t0, tau) {
	return Math.exp(-4.0 * Math.pow((t - t0) / tau, 2));
}


function callback() {
	let t0 = t0_slider.value;
	let tau = tau_slider.value;
	let t = cds.data['t'];
	let s = cds.data['s'];

	for (let i = 0; i < s.length; i++) {
		s[i] = sPulse(t[i], t0, tau);
	}

	cds.change.emit();
}
""",
    "autoactivator_fixed_points": """
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

""",
    "toggle_nullclines": """
function f(x, beta, n) {return beta / (1.0 + Math.pow(x, n));}


function ff(x, betax, nx, betay, ny) {
    return betax / (1.0 + Math.pow(f(x, betay, ny), nx));
}


function rootFun(x, betax, nx, betay, ny) {
    return x - ff(x, betax, nx, betay, ny);
}


function derivff(x, betax, nx, betay, ny) {
    let fy = betay / (1.0 + Math.pow(x, ny));
    let fynx = Math.pow(fy, nx);

    let num = nx * ny * Math.pow(x, ny - 1.0) * betax * fynx * fy;
    let denom = Math.pow(betay * (1.0 + fynx), 2);

    return num / denom;
}


function derivRootFun(x, betax, nx, betay, ny) {
    return 1.0 - derivff(x, betax, nx, betay, ny);
}


function leftRoot(betax, nx, betay, ny, tol=1e-8, maxIters=200) {
    let args = [betax, nx, betay, ny];

    return newtonSolve(
        0.0,
        rootFun,
        derivRootFun,
        args=args,
        tol=tol,
        maxIters=maxIters,
    );
}


function rightRoot(betax, nx, betay, ny, tol=1e-8, maxIters=200) {
    let args = [betax, nx, betay, ny];

    return newtonSolve(
        betay,
        rootFun,
        derivRootFun,
        args=args,
        tol=tol,
        maxIters=maxIters,
    );
}


function findRoots(betax, nx, betay, ny, tol=1e-8, nudge=1e-4) {
    let x1 = leftRoot(betax, nx, betay, ny, tol=tol);
    let x3 = rightRoot(betax, nx, betay, ny, tol=tol);
    let args = [betax, nx, betay, ny];

    // If both null, try Brent's method
    if (x1 === null) {
        if (x3 === null) {
            return [brentSolve(
                rootFun,
                0.0,
                betay,
                args=args,
                tol=tol
                )
            ];
        }
        else return [x3];
    }

    if (x3 === null) return [x1];
    if (Math.abs(x1 - x3) < 2.0 * nudge) return [x1];

    // Make sure x1 < x3
    if (x1 > x3) [x1, x3] = [x3, x1];

    // Get the middle root with Brent's method
    let x2 = brentSolve(
        rootFun,
        x1 + nudge,
        x3 - nudge,
        args=args,
        tol=tol
    );

    if (x2 !== null) return [x1, x2, x3];
    else return [x1];
}


function callback() {
    // Extract data from source and sliders
    let x_x = cds.data['x_x'];
    let x_y = cds.data['x_y'];
    let y_x = cds.data['y_x'];
    let y_y = cds.data['y_y'];
    let beta_x = beta_x_slider.value;
    let beta_y = beta_y_slider.value;
    let n = n_slider.value

    // Update nullclines
    for (let i = 0; i < x_y.length; i++) {
        x_x[i] = beta_x / (1 + Math.pow(y_x[i], n));
        y_y[i] = beta_y / (1 + Math.pow(x_y[i], n));
    }

    // Update fixed points
    let xfp = findRoots(beta_x, n, beta_y, n);
    if (xfp === null) {
        cdsStable.data['x'] = [];
        cdsStable.data['y'] = [];
        cdsUnstable.data['x'] = [];
        cdsUnstable.data['y'] = [];
    }
    else if (xfp.length === 1) {
        cdsStable.data['x'] = xfp;
        cdsStable.data['y'] = [f(xfp[0], beta_y, n)];
        cdsUnstable.data['x'] = [];
        cdsUnstable.data['y'] = [];
    }
    else {
        cdsStable.data['x'] = [xfp[0], xfp[2]];
        cdsStable.data['y'] = [f(xfp[0], beta_y, n), f(xfp[2], beta_y, n)];
        cdsUnstable.data['x'] = [xfp[1]];
        cdsUnstable.data['y'] = [f(xfp[1], beta_y, n)];
    }

    // Emit changes
    cds.change.emit();
    cdsStable.change.emit();
    cdsUnstable.change.emit();
}
""",
}
