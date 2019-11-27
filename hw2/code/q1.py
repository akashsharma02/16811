import numpy as np
import matplotlib.pyplot as plt


def get_coefficients(x, y):
    '''
    Calculates the newton coefficients of the polynomial required for the divided differences method

    Arguments: x -- numpy array containing the input points (x)
               y -- numpy array containing the function output points f(x)

    Returns:   a -- numpy array containing the coefficients
    '''
    n = len(x)
    x = np.asarray(x); y = np.asarray(y)

    x.astype(float)
    y.astype(float)

    F = np.zeros((n, n), dtype = 'float')

    # Fill the first row of the table with output values
    F[:, 0] = y

    # first element a_0 is y_0
    for i in range(1, n):
        for j in range(n-i):
            F[j][i] = (F[j][i-1] - F[j+1][i-1])/(x[j] - x[j+i])


    # According to the formula we only need the first row of the coefficients to compute the polynomial
    a = F[0, :]
    return a


def newton_evaluate(a, x, node):

    x = np.asarray(x); x.astype(float)
    n = len(a) - 1
    temp = a[n]

    # Calculate the newton's polynomial in the nested form from higher order to lower order
    for i in range(n-1, -1, -1):
        temp = temp*(node - x[i]) + a[i]

    return temp

if __name__ == "__main__":
    # 1(b)
    print("##########################################")
    print(" Part 1(b)")
    print("##########################################")
    x = [0, 1/8, 1/4, 1/2, 3/4, 1]
    y = np.exp(np.array(np.negative(x)))
    node = 1/3
    a = get_coefficients(x, y)
    answer = newton_evaluate(a, x, node)
    print("Interpolated value at {} is {}\n\n".format(round(node, 11), round(answer, 11)))

    # 1(c)
    f = lambda x: 1/(1+16*(x**2))
    print("##########################################")
    print(" Part 1(c)")
    print("##########################################")
    node = 0.05
    actual_fx = f(node)
    print("Actual value of f(x) at 0.05 is {}\n".format(actual_fx))
    for n in [2, 4, 40]:
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        for i in range(0, n+1):
            x[i] = i*(2/n) - 1
            y[i] = f(x[i])
        a = get_coefficients(x, y)
        answer = newton_evaluate(a, x, node)
        print("Interpolated value at {} is {}".format(round(node, 5), round(answer, 5)))
        print("Error for n = {} is {}\n".format(n,  round(abs(round(actual_fx, 5) - answer), 5) ))


    print("##########################################")
    print(" Part 1(d)")
    print("##########################################")
    f = lambda x: 1/(1+16*(x**2))
    x = np.linspace(-1, 1, num=1000)
    y = f(x)
    poly_samples = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
    for n in poly_samples:
        # Take only the first n samples of 1000 samples in -1 to 1 to interpolate the polynomial
        x_poly_samples = np.linspace(-1, 1, num=n+1)
        y_poly_samples = f(x_poly_samples)
        a = get_coefficients(x_poly_samples, y_poly_samples)

        error_max = 0
        for x_i in x:
            answer = newton_evaluate(a, x_poly_samples, x_i)
            error = abs(f(x_i) - answer)
            error_max = round(max(error_max, error), 5)
        print("Maximum error for n = {} is {}".format(n, error_max))

    '''
    n = int(input("Enter the number of points to interpolate:"))
    x = np.linspace(-1, 1, n)
    y = np.zeros(n)
    print(x.shape, y.shape)
    for idx, x_val in enumerate(x):
        y[idx] = 1/(1 + 16*(x_val**2))

    node =
    a = get_coefficients(x, y)
    answer = newton_evaluate(a, x, node)
    print("Interpolated value at {} is {}".format(round(node, 11), round(answer, 11)))
    '''
