import numpy as np
import math
import cmath

def get_dd_coefficients(x, y):
    '''
    Calculates the newton coefficients of the polynomial required for the divided differences method

    Arguments: x -- numpy array containing the input points (x)
               y -- numpy array containing the function output points f(x)

    Returns:   a -- numpy array containing the coefficients
    '''
    n = len(x)
    #x = np.asarray(x, dtype=complex); y = np.asarray(y, dtype=complex)

    #x.astype(complex)
    #y.astype(complex)

    F = np.zeros((n, n), dtype = complex)

    # Fill the first row of the table with output values
    F[:, 0] = y

    # first element a_0 is y_0
    for i in range(1, n):
        for j in range(n-i):
            F[j][i] = (F[j][i-1] - F[j+1][i-1])/(x[j] - x[j+i])


    # we need f[x0], f[x_1, x_2], f[x_0, x_1, x_2]

    a = np.array([F[0,0], F[1, 1], F[0,2]])
    return a

def Muller(f, x_0, x_1, x_2, epsilon):
    x = np.array([x_0, x_1, x_2], dtype=complex)
    roots = []
    while len(f.c) > 1:
        while(True):
            fx = f(x)
            if (0 in fx):
                # if any of the initial guesses are roots return
                root = x[np.where(fx == 0)][0]
                break
            coefficients = get_dd_coefficients(x, f(x))
            a = coefficients[2]
            b = coefficients[1] + coefficients[2]*(x[2] - x[1])
            c = f(x[2])

            val1 = 2*c/(b + cmath.sqrt(b**2 - 4*a*c))
            val2 = 2*c/(b - cmath.sqrt(b**2 - 4*a*c))

            if abs(val1) >= abs(val2):
                x_3 = x[2] - val2
            else:
                x_3 = x[2] - val1

            x = np.append(x[1:], x_3)
            if abs(f(x[2])) < epsilon:
                digit_accuracy = math.floor(-math.log10(epsilon))
                root = round(x[2], digit_accuracy)
                break
        roots.append(root)
        f, _ = np.polydiv(f.c, [1, -root])
        f = np.poly1d(f)
    return roots


if __name__ == '__main__':
    f = np.poly1d([1, -5, 11, -15])
    x_0 = 0; x_1 = 1; x_2 = 2;
    roots = Muller(f, x_0, x_1, x_2, epsilon=0.000005)
    print(roots)
