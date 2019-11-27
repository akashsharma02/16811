import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    """return the function

    :x: TODO
    :y: TODO
    :returns: TODO

    """
    return x**3 + y**3 - 2*x**2 + 3*y**2 - 8

def main():
    """Main function
    :returns: TODO

    """
    #TODO: Plot contour plot for f(x, y)
    x = np.arange(-1.5, 2.5, 0.05)
    y = np.arange(-2.5, 1.5, 0.05)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    # plot contours on values where df/dx = 0, and df/dy = 0
    dx = np.array([0, 4/3])
    dy = np.array([0, -2])
    dX, dY = np.meshgrid(dx, dy)
    c = f(dX, dY)
    c = np.sort(c.ravel())


    fig, ax = plt.subplots()
    contour_plt_outside = plt.contour(X, Y, Z, linewidths=1, cmap='viridis')
    ax.clabel(contour_plt_outside, inline=1, fontsize=9)
    contour_plt = plt.contour(X, Y, Z, c, linewidths=1, cmap='viridis')
    ax.clabel(contour_plt, inline=1, fontsize=9)

    # Use larger step size for gradient
    x = np.arange(-1.5, 2.5, 0.25)
    y = np.arange(-2.5, 1.5, 0.25)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    grad_y, grad_x = np.gradient(Z)
    plt.quiver(X, Y, grad_x, grad_y, width=0.002)

    ax.plot(dX, dY, 'b*')
    ax.set_title('Contour plot')
    plt.show()

if __name__ == "__main__":
    main()
