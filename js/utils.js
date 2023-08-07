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


function isClose(x, y, epsilon) {
    /*
     * Return True if |x - y| are within epsilon of each other.
     */
    if (epsilon == null) {
        epsilon = 0.0000001;
    }
    return Math.abs(x - y) < epsilon;
}