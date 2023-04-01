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
