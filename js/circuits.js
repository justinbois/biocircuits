/** 
 * Right-hand side of the Lotka-Volterra system
 * @param {array} xy - 2-array with values for hares and foxes
 * @param {float} t - time point (not used, but necessary for us in rkf solver)
 * @param {float} alpha - parameter alpha for Lotka-Volterra model
 * @param {float} beta - parameter beta for Lotka-Volterra model
 * @param {float} gamma - parameter gamma for Lotka-Volterra model
 * @param {float} delta - parameter delta for Lotka-Volterra model
 */
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


function cascade(yz, t, beta, gamma, n_x, n_y, x_fun, x_args) {
	// Unpack
	var [y, z] = yz;

	var x = x_fun(t, ...x_args);

	var dy_dt = beta * act_hill(x, n_x) - y;
	var dz_dy = gamma * (act_hill(y, n_y) - z)

	return [dy_dt, dz_dt];
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