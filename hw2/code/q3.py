import numpy as np
import math
import matplotlib.pyplot as plt

def bisection_method(a, b, f, tolerance):
    " find root of the function upto tolerance level"
    if f(a)*f(b) > 0:
        return None

    c = a
    while(b - a) > tolerance:
        c = (a+b)/2

        if f(c) == 0.0:
            return c
        elif f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return c

def newton_interpolation(f, Df, x0, epsilon, max_iterations):
    '''
    Inputs:
    f : function lambda
    Df : derivative of function lambda
    x0 : initial guess of the root of function
    epsilon : accepted error value for the root
    max_iterations : max iterations to run the root finding procedure
    '''

    xn = x0
    for iteration in range(0, max_iterations):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print("Found solution after {} tries = {}".format(iteration, xn))
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print("Derivative is 0, cannot converge exiting")
            return None
        xn = xn - fxn/Dfxn
    print("Exceeded max iterations = {}".format(max_iterations))
    return None

if __name__ == '__main__':
    f = lambda x: math.tan(x) - x
    df = lambda x: 1/(math.cos(x)**2) - 1
    x0 = 8
    epsilon = 0.0000005
    max_iterations = 1000

    # plot the function
    x = np.arange(np.pi, 4*np.pi, 0.01)
    plt.plot(x, np.tan(x), x, x)
    plt.axis([np.pi, 4*np.pi, -15, 15])
    plt.title('y = tan(x)')
    plt.axhline()
    plt.show()

    # Use bisection method to obtain initial estimate of the roots
    estimate1 = round(bisection_method(3*math.pi/2, 7, f, 0.001), 2)
    estimate2 = round(bisection_method(7, 5*math.pi/2, f, 0.001), 2)
    print(estimate1, estimate2)

    answer1 = newton_interpolation(f, df, estimate1, epsilon, max_iterations)
    answer2 = newton_interpolation(f, df, estimate2, epsilon, max_iterations)
    print(round(answer1, 6), round(answer2, 6))
