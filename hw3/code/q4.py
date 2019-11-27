import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readfile(filename):
    """ Read a file of format [['x', 'y', 'z'],[...].....['x_n', 'y_n', 'z_n']]

    :filename: Name of the file
    :returns: 3xN np array containing the points

    """
    points = []
    with open(filename) as f:
        points = np.array([line.strip().split(' ') for line in f.readlines()], dtype=float)
    #print(points)
    return points


def getPlane(x, y, params):
    """ Generate z for a plane

    :x: TODO
    :y: TODO
    :params: TODO
    :returns: TODO

    """
    matrix = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), np.ones((x.shape[0], 1)) ))
    Z = np.dot(matrix, params).squeeze(-1)
    #Z = x*params[0] + y*params[1] + params[2]
    return Z

def bestFitPlane(filename):
    """ Fit a best plane for clear_table.txt
    :returns: TODO

    """
    # Fit a plane for clear table
    points = readfile(filename)

    # Plot the 3d Points as a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25)

    # Construct matrix A
    A = np.hstack((points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1), np.ones((points.shape[0], 1)) ))
    z = points[:, 2].reshape(-1, 1)
    best_fit_params = np.matmul(np.linalg.pinv(A), z)
    print("Plane = [a = {}, b = {}, c = 1, d = {}]".format(best_fit_params[0], best_fit_params[1], best_fit_params[2]))
    # Calculate average distance of points to plane
    denom = np.sqrt(best_fit_params[0]**2 + best_fit_params[1]**2 + 1)
    distance_matrix = np.hstack((points[:, 0:2].reshape(-1, 2), -points[:, 2].reshape(-1, 1), np.ones((points.shape[0], 1))))
    distances = np.abs(np.dot(distance_matrix, np.insert(best_fit_params, 2, 1)))/denom
    average_distance = np.average(distances)

    # plot plane of best fit
    x = np.arange(min(points[:, 0]), max(points[:, 0]), 0.1)
    y = np.arange(min(points[:, 1]), max(points[:, 1]), 0.1)
    X, Y = np.meshgrid(x, y)
    XX = X.flatten()
    YY = Y.flatten()
    Z = getPlane(XX, YY, best_fit_params)
    XX = XX[(Z >= min(points[:, 2])-0.4) & (Z <= max(points[:, 2])+0.4)]
    YY = YY[(Z >= min(points[:, 2])-0.4) & (Z <= max(points[:, 2])+0.4)]
    Z = Z[(Z >= min(points[:, 2])-0.4) & (Z <= max(points[:, 2])+0.4)]

    # Plot the wireframe
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25)
    ax1.plot_trisurf(XX, YY, Z, color='orange', alpha=0.6)
    plt.show()
    return average_distance

def bestFitPlaneRansac(filename, no_of_planes=1, max_iter=100, tolerance=[0.25], colors=['orange']):
    """Find the best plane fitting the pointset using RANSAC

    :filename, max_iter: TODO
    :returns: TODO

    """
    points = readfile(filename)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25)
    plt.title("Plot of Point cloud")

    planes = []
    ind = []
    plane_smoothness = []
    point_cloud = points
    for plane in range(no_of_planes):
        max_inliers = 0
        plane_best_smoothness = 0
        for iteration in range(0, max_iter):
            # randomly sample 3 points to fit a plane
            sampled_points = point_cloud[np.random.randint(point_cloud.shape[0], size=3), :]
            A_matrix = np.hstack((sampled_points[:, 0].reshape(-1, 1), sampled_points[:, 1].reshape(-1, 1), np.ones((sampled_points.shape[0], 1)) ))
            z = sampled_points[:, 2].reshape(-1, 1)
            fit = np.matmul(np.linalg.pinv(A_matrix), z)

            # get distances of points to plane to calculate inliers
            denom = np.sqrt(fit[0]**2 + fit[1]**2 + 1)
            distance_matrix = np.hstack((point_cloud[:, 0:2].reshape(-1, 2), -point_cloud[:, 2].reshape(-1, 1), np.ones((point_cloud.shape[0], 1))))
            distances = np.abs(np.dot(distance_matrix, np.insert(fit, 2, 1)))/denom
            distance_inliers = np.where(distances <= tolerance[plane], 1, 0)
            distance_outliers = np.where(distances >tolerance[plane], 1, 0)
            inliers = np.sum(distance_inliers)

            # Calculate smoothness of the plane as inliers / Sum(distances from plane)
            smoothness = inliers/np.sum(np.where(distances <= tolerance[plane], distances, 0))
            smoothness *= tolerance[plane]

            # choose the parameters which retains the max number of inliers
            if inliers > max_inliers:
                max_inliers = inliers
                best_fit_params = fit
                ind = np.nonzero(distance_outliers)
                plane_best_smoothness = smoothness
        # retain the outliers to search for next plane
        point_cloud = point_cloud[ind[0], :]
        planes.append(best_fit_params)
        plane_smoothness.append(plane_best_smoothness)

    # plot plane of best fit
    x = np.linspace(min(points[:, 0]-0.5), max(points[:, 0]+0.5), 100)
    y = np.linspace(min(points[:, 1]-0.5), max(points[:, 1]+0.5), 100)
    X, Y = np.meshgrid(x, y)
    XX = X.flatten()
    YY = Y.flatten()
    ax1 = fig.add_subplot(122, projection='3d')
    for idx, plane in enumerate(planes):
        X = XX; Y = YY;
        print("Plane = [a = {}, b = {}, c = [1], d = {}]".format(plane[0], plane[1], plane[2]))
        Z = getPlane(XX, YY, plane)
        X = X[(Z >= min(points[:, 2])-0.2) & (Z <= max(points[:, 2])+0.2)]
        Y = Y[(Z >= min(points[:, 2])-0.2) & (Z <= max(points[:, 2])+0.2)]
        Z = Z[(Z >= min(points[:, 2])-0.2) & (Z <= max(points[:, 2])+0.2)]

        # Plot the wireframe
        ax1.plot_trisurf(X, Y, Z, color=colors[idx], linewidth=0.2, alpha=0.6, label='RANSAC Planes')

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, label='Point Cloud')
    ax1.set_zlim(min(points[:, 2]), max(points[:, 2]))
    plt.title("Plot of the Point clouds and best fitted planes")
    #plt.legend(loc=2)
    plt.show()
    return plane_smoothness


def main():
    """ Main function

    :: TODO
    :returns: TODO

    """
    average_dist1 = bestFitPlane('./clear_table.txt')
    print("Average Distance for Plane 1 for clear_table.txt point cloud: {}\n".format(average_dist1))
    average_dist2 = bestFitPlane('./cluttered_table.txt')
    print("Average Distance for Plane 2 for cluttered table.txt point cloud: {}\n".format(average_dist2))
    smoothness = bestFitPlaneRansac('./cluttered_table.txt', max_iter=200, tolerance=[0.005])
    print("Smoothness = {}\n".format(smoothness))
    colors1=['red', 'orange', 'green', 'yellow']
    smoothness = bestFitPlaneRansac('./clean_hallway.txt', 4, max_iter=400, tolerance=[0.01, 0.01, 0.01, 0.01], colors=colors1)
    print("Smoothness = {}\n".format(smoothness))
    smoothness = bestFitPlaneRansac('./cluttered_hallway.txt', 4, max_iter=1000, tolerance=[0.02, 0.02, 0.02, 0.05], colors=colors1)
    print("Smoothness = {}\n".format(smoothness))
    print("\n")


if __name__ == "__main__":
    main()


