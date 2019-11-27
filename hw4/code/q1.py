import numpy as np
import matplotlib.pyplot as plt

def plot_values(x, y, y_estimated):
    """Plot the table and graph for x, and estimated values

    :x: TODO
    :y: TODO
    :returns: TODO

    """
    fig, axs = plt.subplots(1, 2)
    cell_text = []
    cell_text.append(["%1.2f"% xi for xi in x])
    cell_text.append(["%1.5f"% yi for yi in y])
    cell_text.append(["%1.5f"% yi_est for yi_est in y_estimated])
    cell_text.append(["%1.5f"% error_i for error_i in np.abs(y_estimated - y)])
    cell_text = np.array(cell_text).T
    col_labels = ['Xi', 'True Value', 'Estimated value', 'Error Value']

    axs[0].plot(x, y, color='blue', linewidth=1, label='True value')
    axs[0].plot(x, y_estimated, color='red', linewidth=1, label='Estimate value')
    axs[0].legend()
    table = axs[1].table(cellText=cell_text, colLabels=col_labels,loc='center')
    #table.scale(0.6, 0.6)
    axs[1].axis('off')
    plt.show()

def analytical_sol(x):
    """ Return the computed analytical solution of the ODE

    :x: TODO
    :returns: TODO

    """
    return np.sqrt(2*(x - 1 + 1e-14))

def f(x):
    return 1/x

def euler(f, x, x0, y0, h ):
    """TODO: Docstring for euler.

    :f: TODO
    :x: array of all x values to compute
    :x0: Initial value of x
    :y0: known initial value of y
    :h: step size to compute approximation
    :returns: y

    """
    y = [y0]
    y_i = y0
    for i in range(1, x.shape[-1]):
        y_n1 = y_i + h * f(y_i)
        y.append(y_n1)
        y_i = y_n1
    return np.array(y)

def runge_kutta4(f, x, x0, y0, h):
    """Evaluate and return y for ODE using RK4 integration method

    :f: TODO
    :x: TODO
    :x0: TODO
    :y0: TODO
    :h: TODO
    :returns: TODO

    """
    y = [y0]
    y_i = y0
    for i in range(1, x.shape[-1]):
        k1 = h*f(y_i)
        k2 = h*f(y_i + k1/2)
        k3 = h*f(y_i + k2/2)
        k4 = h*f(y_i + k3)
        y_n1 = y_i + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        y.append(y_n1)
        y_i = y_n1
    return np.array(y)

def adams_bashforth4(f, x, x0, y0, h):
    """Solve ODE and return y using adams

    :f: TODO
    :x: TODO
    :x0: TODO
    :y0: TODO
    :h: TODO
    :returns: TODO

    """
    y = [y0]
    y_i = y0
    f4 = f(1.51657508881031);
    f3 = f(1.48323969741913);
    f2 = f(1.44913767461894);
    f1 = f(1.4142135623731);
    for i in range(1, x.shape[-1]):
        y_n1 = y_i + h/24 * (55*f1 - 59*f2 +37*f3 -9*f4)
        y.append(y_n1)
        f4, f3, f2, f1 = f3, f2, f1, f(y_n1)
        y_i = y_n1
    return np.array(y)
def main():
    """Main function

    :errors: tuple containing average errors from euler's method, RK4 and Adams Bashforth methods

    """
    # Implement Euler's method for q1.2
    x0 = 2
    y0 = 1.414214
    h  = 0.05
    cell_text = []
    row_labels = ['x', 'y_true', 'y_euler', 'y_rk4', 'y_ab4']

    x = np.arange(2, 1-h, -h)
    print(x)
    y_true = np.round(analytical_sol(x), 5)
    print(y_true)

    y_euler = euler(f, x, x0, y0, -h)
    print(y_euler)
    avg_error_euler = np.sum(np.abs(y_euler - y_true))/y_true.shape[-1]
    print(avg_error_euler)
    plot_values(x, y_true, y_euler)

    y_rk4 = runge_kutta4(f, x, x0, y0, -h)
    avg_error_rk4 = np.sum(np.abs(y_rk4 - y_true))/y_true.shape[-1]
    print(avg_error_rk4)
    plot_values(x, y_true, y_rk4)

    y_ab4 = adams_bashforth4(f, x, x0, y0, -h)
    avg_error_ab4 = np.sum(np.abs(y_ab4 - y_true))/y_true.shape[-1]
    print(avg_error_ab4)
    plot_values(x, y_true, y_ab4)


if __name__ == "__main__":
    main()
