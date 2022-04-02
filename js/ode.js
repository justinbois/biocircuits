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
    enforceNonnegative,
    debugMode,
) {
    // Set up return variables
    let tSol = [timePoints[0]];
    let t = timePoints[0];
    let iMax = timePoints.length;
    let y = [initialCondition];
    let y0 = initialCondition;
    let i = 1;
    let nDeadSteps = 0;
    let deadStep = false;

    // DEBUG
    let nSteps = 0;
    // END EDEBUG

    // Default parameters
    let h;
    if (dt === undefined) h = timePoints[1] - timePoints[0];
    else h = dt;

    if (tol === undefined) tol = 1e-7;
    if (relStepTol === undefined) relStepTol = 0.0;
    if (sBounds === undefined) sBounds = [0.1, 10.0];
    if (hMin === undefined) hMin = 0.0;
    if (enforceNonnegative === undefined) enforceNonnegative = true;
    if (maxDeadSteps === undefined) maxDeadSteps = 10;
    if (debugMode === undefined) debugMode = false;

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
    if (debugMode) console.log(nSteps);
    // END DEBUG

    let yInterp;
    if (nDeadSteps == maxDeadSteps) {
    	yInterp = nanArray(initialCondition.length, iMax);
    }
    else yInterp = interpolateSolution(timePoints, tSol, transpose(y));

    return yInterp;
}


function rkf45Step(f, y, t, args, h, tol, relStepTol, sBounds, hMin) {
    let k1 = svMult(h, f(y, t, ...args));

    let y2 = svMultAdd([0.25, 1.0], [k1, y]);
    let k2 = svMult(h, f(y2, t + 0.25 * h, ...args));

    let y3 = svMultAdd([0.09375, 0.28125, 1.0], [k1, k2, y]);
    let k3 = svMult(h, f(y3, t + 0.375 * h, ...args));

    let y4 = svMultAdd(
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 1.0],
        [k1, k2, k3, y]
    );
    let k4 = svMult(h, f(y4, t + (12.0 * h) / 13.0, ...args));

    let y5 = svMultAdd(
        [
            8341.0 / 4104.0,
            -32832.0 / 4104.0,
            29440.0 / 4104.0,
            -845.0 / 4104.0,
            1.0,
        ],
        [k1, k2, k3, k4, y]
    );
    let k5 = svMult(h, f(y5, t + h, ...args));

    let y6 = svMultAdd(
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
    let k6 = svMult(h, f(y6, t + h / 2.0, ...args));

    // Calculate new step
    let yNew = svMultAdd(
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
	let relChangeStep = norm(vectorAdd(yNew, svMult(-1.0, y))) / norm(yNew);

    // Calculate error (note that k2's contribution to the error is zero)
    let errorVector = svMultAdd(
        [
            209.0 / 75240.0,
            -2252.8 / 75240.0,
            -2197.0 / 75240.0,
            1504.8 / 75240.0,
            2736.0 / 75240.0,
        ],
        [k1, k3, k4, k5, k6]
    );
    let error = Math.max(...absVector(errorVector));

    // Either don't take a step or use the RK4 step
    let deadStep;
    if (error < tol || relChangeStep < relStepTol || h <= hMin) {
        t += h;
        deadStep = false;
    } else {
        yNew = y;
        deadStep = true;
    }

    // Compute scaling for new step size
    let s;
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


function dydtIMEX(y, t, f, cfun, Afun, fArgs, cfunArgs, AfunArgs, diagonalA) {
    /*
     * Right hand side of ODEs for initializing IMEX method with RKF.
     */

    n = y.length;
    let rhs = zeros(n);

    let A = Afun(t, ...AfunArgs);
    let c = cfun(t, ...cfunArgs);

    // Linear part
    let nonConstantLinear = diagonalA
        ? elementwiseVectorMult(A, y)
        : mvMult(A, y, diagonalA);
    let linearPart = vectorAdd(nonConstantLinear, c);

    // Nonlinear part
    let nonlinearPart = f(y, t, ...fArgs);

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

    let invk = 1.0 / k;
    let b = vectorAdd(
        svMult(0.5, c),
        svMult(invk, u),
        svMult(1.0 + omega / 2.0, f1),
        svMult(-omega / 2.0, f0),
        svMult(0.5, g1)
    );

    if (diagonalA) {
        let Aaug = svAdd(invk, svMult(-0.5, A));
        let result = elementwiseVectorDivide(b, Aaug);
    } else {
        let n = A.length;
        let Aaug = smMult(-0.5, A);
        for (i = 0; i < n; i++) {
            Aaug[i][i] += invk;
        }
        let result = solve(Aaug, b);
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
    let mult =
        Math.pow(relChange[1] / relChangeStep, kP) *
        Math.pow(tol / relChangeStep, kI) *
        Math.pow(Math.pow(relChange[0], 2) / relChange[1] / relChangeStep, kD);
    if (mult > sBounds[1]) mult = sBounds[1];
    else if (mult < sBounds[0]) mult = sBounds[0];

    let newk = mult * k;

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

    let mult = tol / relChangeStep;
    if (mult < sBounds[0]) mult = sBounds[0];

    let newk = mult * k;
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
    let rkf45TimePoints = [
        timePoints[0],
        timePoints[0] + k0,
        timePoints[0] + 2.0 * k0,
    ];

    let args = [f, cfun, Afun, fArgs, cfunArgs, AfunArgs, diagonalA];
    let yRKF = rkf45(
        dydtIMEX,
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
    let tSol = [timePoints[0]];
    let iMax = timePoints.length;
    let y = [initialCondition];
    let k = 2.0 * k0;
    let newk;
    let t = rkf45TimePoints[2];
    let y0 = yRKF[2];
    let i = 1;
    let nDeadSteps = 0;
    let deadStep = false;
    let c = cfun(t, ...cfunArgs);
    let A = Afun(t, ...AfunArgs);
    let f0 = f(initialCondition, timePoints[0], ...fArgs);
    let f1 = f(y0, t, ...fArgs);
    let g1 = vectorAdd(c, mvMult(A, y0, diagonalA));
    let omega = 1.0;
    let yStep;
    let relChangeStep;
    let relTol = tol * (1.0 + tolBuffer);
    let relChange = [
        norm(vectorAdd(y0, svMult(-1.0, yRKF[1]))) / norm(y0),
        norm(vectorAdd(yRKF[1], svMult(-1.0, initialCondition))) /
            norm(yRKF[1]),
    ];

    // DEBUG
    let nSteps = 3;
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
    // END DEBUG

    if (nDeadSteps == maxDeadSteps) {
        return nanArray(initialCondition, iMax);
    }
    let yInterp = interpolateSolution(timePoints, tSol, transpose(y));

    return yInterp;
}


function interpolate1d(x, xs, ys) {
    let y2s = naturalSplineSecondDerivs(xs, ys);

    let yInterp = x.map(function (xVal) {
        return splineEvaluate(xVal, xs, ys, y2s);
    });

    return yInterp;
}


function interpolateSolution(timePoints, t, y) {
    // Interpolate each row of y
    let yInterp = y.map(function (yi) {
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

    let n = xs.length;

    // Storage used in tridiagonal solve
    let u = zeros(n);

    // Return value
    let y2s = zeros(n);

    // Solve trigiadonal matrix by decomposition
    for (let i = 1; i < n - 1; i++) {
        let fracInterval = (xs[i] - xs[i - 1]) / (xs[i + 1] - xs[i - 1]);
        let p = fracInterval * y2s[i - 1] + 2.0;
        y2s[i] = (fracInterval - 1.0) / p;
        u[i] =
            (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) -
            (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]);
        u[i] =
            ((6.0 * u[i]) / (xs[i + 1] - xs[i - 1]) - fracInterval * u[i - 1]) /
            p;
    }

    // Tridiagonal solve back substitution
    for (let k = n - 2; k >= 0; k--) {
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
    let n = xs.length;

    // Indices bracketing where x is
    let lowInd = 0;
    let highInd = n - 1;

    // Perform bisection search to find index of x
    while (highInd - lowInd > 1) {
        let i = (highInd + lowInd) >> 1;
        if (xs[i] > x) {
            highInd = i;
        } else {
            lowInd = i;
        }
    }
    let h = xs[highInd] - xs[lowInd];
    let a = (xs[highInd] - x) / h;
    let b = (x - xs[lowInd]) / h;

    let y = a * ys[lowInd] + b * ys[highInd];
    y +=
        (((Math.pow(a, 3) - a) * y2s[lowInd] + (Math.pow(b, 3) - b) * y2s[highInd]) * Math.pow(h, 2)) /
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
// let lv = lotkaVolterraIMEX(1.0, 2.0, 3.0, 4.0);
// let sol = vsimex(lv.f, lv.cfun, lv.Afun, [1.0, 3.0], linspace(0.0, 20.0, 200), [], [], [], lv.diagonalA)
