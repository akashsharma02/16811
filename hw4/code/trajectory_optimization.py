import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

waypoints = 300
N = 101
OBST = np.array([[20, 30], [60, 40], [70, 85]])
epsilon = np.array([[25], [20], [30]])

obs_cost = np.zeros((N, N))
for i in range(OBST.shape[0]):
    t = np.ones((N, N))
    t[OBST[i, 0], OBST[i, 1]] = 0
    t_cost = distance_transform_edt(t)
    t_cost[t_cost > epsilon[i]] = epsilon[i]
    t_cost = 1 / (2 * epsilon[i]) * (t_cost - epsilon[i])**2
    obs_cost = obs_cost + t_cost

gx, gy = np.gradient(obs_cost)

SX = 10
SY = 10
GX = 90
GY = 90

traj = np.zeros((2, waypoints))
traj[0, 0] = SX
traj[1, 0] = SY
dist_x = GX-SX
dist_y = GY-SY
for i in range(1, waypoints):
    traj[0, i] = traj[0, i-1] + dist_x/(waypoints-1)
    traj[1, i] = traj[1, i-1] + dist_y/(waypoints-1)

path_init = traj.T
tt = path_init.shape[0]
path_init_values = np.zeros((tt, 1))
for i in range(tt):
    path_init_values[i] = obs_cost[int(np.floor(path_init[i, 0])), int(np.floor(path_init[i, 1]))]

# # Plot 2D
# plt.imshow(obs_cost.T)
# plt.plot(path_init[:, 0], path_init[:, 1], 'ro')

# # Plot 3D
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(111, projection='3d')
# xx, yy = np.meshgrid(range(N), range(N))
# ax3d.plot_surface(xx, yy, obs_cost, cmap=plt.get_cmap('coolwarm'), alpha=0.8)
# ax3d.scatter(path_init[:, 1], path_init[:, 0], path_init_values, s=20, c='r')

plt.show()

# Optimization code starts from here

path = path_init.copy()
# interpolate the gradient of the path from the gx and gy
def plot_path(obs_cost, path, N, iteration_no):
    """Plot 2D and 3D path on the given obstacle cost function

    :obs_cost: TODO
    :path: TODO
    :returns: TODO

    """
    fig2d = plt.figure()
    plt.imshow(obs_cost.T)
    plt.plot(path[:, 0], path[:, 1], 'r.', markersize=5)
    plt.title('2D Plot of path after ' + str(counter) + ' iterations')

    # Plot 3D
    path_values = np.zeros((path.shape[0], 1))
    for i in range(path.shape[0]):
        path_values[i] = obs_cost[int(np.floor(path[i, 0])), int(np.floor(path[i, 1]))]

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(range(N), range(N))
    ax3d.plot_surface(xx, yy, obs_cost, cmap=plt.get_cmap('coolwarm'), alpha=0.8)
    ax3d.scatter(path[:, 1], path[:, 0], path_values, s=2, c='r')
    plt.title('3D Plot of path after ' + str(counter) + ' iterations')
    plt.show()

# Gradient descent the points until they converge
threshold = 0.5
x = np.arange(0, N, 1, dtype='float64')

spline_gx = RectBivariateSpline(x, x, gx)
spline_gy = RectBivariateSpline(x, x, gy)

counter = 0

# Interpolate the gradients at fractional coordinates
# Q 6(a)
gx_path = spline_gx.ev(path[:, 0], path[:, 1])
gy_path = spline_gy.ev(path[:, 0], path[:, 1])

while np.linalg.norm(gx_path) > threshold and np.linalg.norm(gy_path) > threshold:
    gx_path = spline_gx.ev(path[:, 0], path[:, 1])
    gy_path = spline_gy.ev(path[:, 0], path[:, 1])
    # Plot the path after one iteration of gradient descent with alpha = 0.1
    path[1:-1, :] = path[1:-1, :] - np.array([0.1*gx_path[1:-1], 0.1*gy_path[1:-1]]).T
    if counter == 1:
        plot_path(obs_cost, path, N, counter)
    counter += 1

plot_path(obs_cost, path,N, counter)



# Q 6(b)
# Add another minimizer which reduces  hx = 0.5| grad_x_i - grad_x_{i-1} |
counter = 0
path = path_init.copy()
gx_path = spline_gx.ev(path[:, 0], path[:, 1])
gy_path = spline_gy.ev(path[:, 0], path[:, 1])
cost_x = gx_path
cost_y = gy_path
while counter < 501:
    gx_path = spline_gx.ev(path[:, 0], path[:, 1])
    gy_path = spline_gy.ev(path[:, 0], path[:, 1])

    # Vectorize hx and hy computation
    penalty = (path[1:] - path[:-1])
    print(penalty.shape)

    cost = 0.8*np.stack((gx_path[1:-1], gy_path[1:-1]), -1) + 4*penalty[:-1]

    path[1:-1, :] = path[1:-1, :] - 0.1*cost
    path[1:-1, :] = np.where(path[1:-1, :] < [N, N], path[1:-1, :], N-1)
    print(counter)
    if counter in [100, 200, 500]:
        plot_path(obs_cost, path, N, counter)
    counter += 1


# Q 6(c)
# Add another term in the minimizer to account for the previous path element
counter = 0
path = path_init.copy()
gx_path = spline_gx.ev(path[:, 0], path[:, 1])
gy_path = spline_gy.ev(path[:, 0], path[:, 1])
cost_x = gx_path
cost_y = gy_path
while counter < 5001:
    gx_path = spline_gx.ev(path[:, 0], path[:, 1])
    gy_path = spline_gy.ev(path[:, 0], path[:, 1])

    # Vectorize hx and hy computation
    penalty = (-path[2:] + 2*path[1:-1] - path[:-2])
    cost = 0.8*np.stack((gx_path[1:-1], gy_path[1:-1]), -1) + 4*penalty

    path[1:-1, :] = path[1:-1, :] - 0.1*cost
    path[1:-1, :] = np.where(path[1:-1, :] < [N, N], path[1:-1, :], N-1)
    print(counter)
    if counter in [100, 5000]:
        plot_path(obs_cost, path, N, counter)
    counter += 1

