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

