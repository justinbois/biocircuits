function sensitivity(a0, b0, Kd) {
	let b = a0 - b0 - Kd;
	let discrim = b * b + 4 * a0 * Kd;
	let sqrtDiscrim = Math.sqrt(discrim);

	return a0 * (1 - b / sqrtDiscrim) / (sqrtDiscrim - b);
}


function callback() {
    let b0 = Math.pow(10, b0Slider.value);
    let KdVals = [0.001, 0.01, 0.1, 1.0, 10.0];        
    let a0 = cds.data['a0'];
    
    for (let i = 0; i < KdVals.length; i++) {
    	let ind = 's' + i.toString();
    	for (let j = 0; j < a0.length; j++) {
    		cds.data[ind][j] = sensitivity(a0[j], b0, KdVals[i]);
    	}
    }

    span.location = b0;

    cds.change.emit();
}
