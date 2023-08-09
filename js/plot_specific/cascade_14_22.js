/** 
 * Return steady-state fraction of fully-phorsophorlated molecules in a multi-step phosphorylation reaction.
 * @param {float} s - value of s to compute fn(s)
 * @param {float} n - number of phosphorylation sites
 */
function fMultiPhos(s, n) {
	let res;

	if (Number.isFinite(n)) {
		if (isClose(s, 1.0)) {
			res = 1.0 / (1.0 + n);
		}
		else {
			res = Math.pow(s, n) * (1.0 - s) / (1.0 - Math.pow(s, n + 1));
		}
	}
	else {
		res = s > 1.0 ? (s - 1.0) / s : 0.0;
	}

	return res;
}


/** 
 * Return derivative of steady-state fraction of fully-phorsophorlated molecules in a multi-step phosphorylation reaction.
 * @param {float} s - value of s to compute dfn(s)/ds.
 * @param {float} n - number of phosphorylation sites
 */
function fPrimeMultiPhos(s, n) {
	let res;

	if (Number.isFinite(n)) {
		if (isClose(s, 1.0)) {
			res = n / 2.0 / (1.0 + n);
		}
		else {
			let numerator = Math.pow(s, n - 1) * (n * (1.0 - s) - s * (1.0 - Math.pow(s, n)));
			let denominator = Math.pow((1.0 - Math.pow(s, n + 1)), 2);
			res = numerator / denominator;
		}
	}
	else {
		if (isClose(s, 1.0)) res = 0.5;
		else if (s < 1.0) res = 0.0;
		else res = 1.0 / Math.pow(s, 2);
	}

	return res;
}


/** 
 * Compute the steady-state transfer function of a cascade polyphosphorylation signaling system.
 * @param {float} s - values of input signal
 * @param {array} xTot - total concentration of each kinase
 * @param {array} n - number of phosphorylation sites in each kinase
 */
function transferFunction(s, xTot, n) {    
    // Innermost of nested function calls
    let res = xTot[0] * fMultiPhos(s, n[0]);

    for (let i = 1; i < xTot.length; i++) {
    	res = xTot[i] * fMultiPhos(res, n[i]);
    }

    return res;
}


/** 
 * Compute the log of the steady-state transfer function of a cascade polyphosphorylation signaling system.
 * @param {float} s - values of input signal
 * @param {array} xTot - total concentration of each kinase
 * @param {array} n - number of phosphorylation sites in each kinase
 */
function logTransferFunction(s, xTot, n) {
	let res;
	let tf = transferFunction(s, xTot, n);

	if (tf > 0.0) {
		res = Math.log(tf);
	}
	else if (n.every(x => Number.isFinite(x))) {
		// Approximate calculation for small s
		let logxTot = xTot.map(Math.log);
		let logS = Math.log(s);

		let nProd = [];
		for (let j = 0; j < n.length; j++) {
			nProd.push(n.slice(j + 1).reduce((a, b) => a * b, 1));
		}

		res = n.reduce((a, b) => a * b, 1) * logS;
		for (let j = 0; j < n.length; j++) {
			res += nProd[j] * logxTot[j];
		}
	}
	else {
		res = -Infinity;
	}

    return res;
}


/** 
 * Compute the derivative of the steady-state transfer function of a cascade polyphosphorylation signaling system.
 * @param {float} s - values of input signal
 * @param {array} xTot - total concentration of each kinase
 * @param {array} n - number of phosphorylation sites in each kinase
 */
function logTfDeriv(s, xTot, n) {
	let res;
	if (!n.every(x => Number.isFinite(x)) && s < 1.0) {
		res = -Infinity;
	}
	else {
		// Arguments for transfer function
		let args = [s];
		for (let i = 0; i < xTot.length - 1; i++) {
			args.push(xTot[i] * fMultiPhos(args[i], n[i]));
		} 

		if (!args.every(x => x > 0.0)) {
			let sumLogn = n.map(x => Math.log(x)).reduce((a, b) => a + b, 0);
			res = sumLogn - Math.log(s) + logTransferFunction(s, xTot, n);
		}
		else {
			res = xTot.map(x => Math.log(x)).reduce((a, b) => a + b, 0);
			for (let i = 0; i < n.length; i++) {
				res += Math.log(fPrimeMultiPhos(args[i], n[i]))
			}
		}
	}

	return res;
}


/** 
 * Callback for updating ColumnDataSource with recomputed transfer functions and characterizations thereof.
 */
function callback() {
	// Pull values from sliders and deactivate as necessary
	let xTot14;
	let xTot22 = [Math.pow(10.0, xTot1_22Slider.value), Math.pow(10.0, xTot2_22Slider.value)];
	if (lockXtot14Toggle.active) {
		xTot14 = [xTot22[0] + xTot22[1]];
		xTot1_14Slider.value = Math.log10(xTot22[0] + xTot22[1]);
		xTot1_14Slider.disabled = true;
	}
	else {
		xTot14 = [Math.pow(10.0, xTot1_14Slider.value)];
		xTot1_14Slider.disabled = false;
	}

	// Useful to have around
	let s = cds.data['s'];
	let logS = s.map(x => Math.log(x));

	// Intermediate variables in calculations
	let logTf;
	let logDeriv;
	let n;

	// Baseline
	for (let i = 0; i < s.length; i++) {
		logTf = logTransferFunction(s[i], xTot14, [1]);
		logDeriv = logTfDeriv(s[i], xTot14, [1]);
		cds.data['tf_baseline'][i] = Math.exp(logTf);
		cds.data['deriv_baseline'][i] = Math.exp(logDeriv);
		cds.data['gain_baseline'][i] = Math.exp(logTf - logS[i]);
		cds.data['sens_baseline'][i] = Math.exp(logS[i] - logTf + logDeriv);
	}


	// 14 and 22 circuit
	for (let circuit of ['14', '22']) {
		let tfCol = 'tf_' + circuit;
		let derivCol = 'deriv_' + circuit;
		let gainCol = 'gain_' + circuit;
		let sensCol = 'sens_' + circuit;
		let xTot = circuit == '14' ? xTot14 : xTot22;
		let n = circuit == '14' ? [4] : [2, 2];
		for (let i = 0; i < s.length; i++) {
			logTf = logTransferFunction(s[i], xTot, n);
			logDeriv = logTfDeriv(s[i], xTot, n);
			cds.data[tfCol][i] = Math.exp(logTf);
			cds.data[derivCol][i] = Math.exp(logDeriv);
			cds.data[gainCol][i] = Math.exp(logTf - logS[i]);
			if (logTf < -300) {
				cds.data[sensCol][i] = n.reduce((a, b) => a * b, 1);
			}
			else {
				cds.data[sensCol][i] = Math.exp(logS[i] - logTf + logDeriv);
			}
		}
	}

	cds.change.emit();
}
