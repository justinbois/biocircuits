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