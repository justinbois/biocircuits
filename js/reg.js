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