import numpy as np
import matplotlib.pyplot as plt

x = lambda x: x
x2 = lambda x: x**2


def approximate_poly(x, y, order):
    """ Approximate nth order polynomial for given data

    :x: TODO
    :y: TODO
    :order: TODO
    :returns: TODO

    """
    phi = []
    for i in range(0, order+1):
        phi_order = x**i
        phi.append(phi_order)
    phi = np.asarray(phi)
    phi = np.squeeze(phi, axis=2).T
    # Take pseudo inverse of phi matrix to obtain the normal projection of f(x)
    phi_inv = np.linalg.pinv(phi)
    c = np.round(np.matmul(phi_inv, y), 3)
    return c


if __name__ == '__main__':
    with open('./problem2.txt') as f:
        points = np.array(f.readline().strip().split(' '))
    y = points.astype(float)

    plt.plot(np.arange(0, len(y)), y)
    plt.show()

    # we observe from plot that f is non-differentiable at x = 3
    # therefore we can represent the y as a piecewise function
    x = np.arange(0, len(y))/10.0

    # Second order differential of the curve shows that there is a rapid change at x = 3.0
    y_grad = np.gradient(y, 0.1)
    y_grad2 = np.gradient(y_grad, 0.1)
    plt.plot(x, y_grad2)
    plt.axvline(x=3.0, linewidth=0.8, color='black')
    plt.show()

    # divide the y into two piecewise curves
    indlow = np.argwhere(x <= 3.0)
    indhigh = np.argwhere(x > 3.0)
    xlow = x[indlow]
    xhigh = x[indhigh]
    ylow = y[indlow]
    yhigh = y[indhigh]

    # for x < 3:
    plt.plot(xlow, ylow)
    #plt.show()
    c = approximate_poly(xlow, ylow, 3)
    print("c = {}".format(c))

    # for x > 3
    c1 = approximate_poly(xhigh, yhigh, 3)
    print("c = {}".format(c1))
