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