import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
def readFile(filepath):
    x = []
    y = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                x.append(np.array(line.strip().split(' '), float))
            else:
                y.append(np.array(line.strip().split(' '), float))
    x = np.array(x)
    y = np.array(y)
    return x, y

def classifyPoints(x, y):
    right = []
    left = []
    for path_x, path_y in zip(x, y):
        #print("PathX = {}".format(path_x))
        #print("PathY = {}".format(path_y))
        #print("PathX[0] = {}, PathY[0] = {}".format(path_x[0], path_y[0]))
        if path_x[0] > path_y[0]:
            right.append(list(zip(path_x, path_y)))
        else:
            left.append(list(zip(path_x, path_y)))
    return np.array(right), np.array(left)


def sortPaths(point, paths):
    start_points = paths[:, 0, :]
    #print(start_points)
    distances = (point[0] - start_points[:, 0])**2 + (point[1] - start_points[:, 1])**2
    indices = np.argsort(distances)
    #print(indices)
    #for i in indices:
    #    print("index i path = ", paths[i, :, :])
    paths = paths[indices, :, :]
    print(paths.shape)
    #print(paths)
    return paths

def calcAlpha(start_point, triangle_vertices):
    if len(triangle_vertices) != 3:
        print("wrong vertices sent!")
        return
    print(triangle_vertices)
    distA = scipy.spatial.distance.euclidean(start_point, triangle_vertices[0])
    distB = scipy.spatial.distance.euclidean(start_point, triangle_vertices[1])
    distC = scipy.spatial.distance.euclidean(start_point, triangle_vertices[2])
    print(distA, distB, distC)

    alpha_i = 1/distA; alpha_j = 1/distB; alpha_k = 1/distC;
    normalize_val = alpha_i + alpha_j + alpha_k
    alpha_i /= normalize_val; alpha_j /= normalize_val; alpha_k /= normalize_val

    return alpha_i, alpha_j, alpha_k


def getNewPath(triangle_paths, alpha_i, alpha_j, alpha_k):
    print(triangle_paths)
    newPath = triangle_paths[0, :, :]*alpha_i + triangle_paths[1, :, :]*alpha_j + triangle_paths[2, :, :]*alpha_k
    return newPath

def plotPath(idx, triangle_paths, fx, fy):

    timesteps = np.arange(0, 49, 0.01)

    # Plot ring of fire
    x = np.linspace(5-1.5, 5+1.5, 100)
    y = np.linspace(5-1.5, 5+1.5, 100)
    X, Y = np.meshgrid(x, y)
    F = (X - 5)**2 + (Y - 5)**2 - (1.5)**2
    plt.contour(X, Y, F, [0])
    #circle = plt.Circle((5, 5), 1.5, color='r')
    plt.plot(triangle_paths[0, :, 0], triangle_paths[0, :, 1], linewidth=0.25, color='skyblue', label='triangle_path'+ str(idx))
    plt.plot(triangle_paths[1, :, 0], triangle_paths[1, :, 1], linewidth=0.25, color='skyblue', label='triangle_path' + str(idx))
    plt.plot(triangle_paths[2, :, 0], triangle_paths[2, :, 1], linewidth=0.25, color='skyblue', label='triangle_path'+ str(idx))
    plt.plot(fx(timesteps), fy(timesteps), linewidth=0.5, color='red', label='interpolated_path'+str(idx))
    plt.legend(loc="upper left", prop={'size': 6})


def isInTriangle(point, A, B, C):
    print(A, B, C)
    if np.equal(A, B).all() or np.equal(B, C).all() or np.equal(C, A).all():
        return False
    v0 = B - A
    v1 = C - A
    v2 = point - A
    print("v0 = {}, v1 = {}, v2 = {}".format(v0, v1, v2))

    dot_00 = np.dot(v0, v0)
    dot_01 = np.dot(v0, v1)
    dot_02 = np.dot(v0, v2)
    dot_11 = np.dot(v1, v1)
    dot_21 = np.dot(v2, v1)

    A_matrix = np.array([[dot_00, dot_01], [dot_01, dot_11]])
    b = np.array([[dot_02],[dot_21]])

    U = np.linalg.solve(A_matrix, b)
    return (U[0] > 0.0 and U[1] > 0.0 and U[0] + U[1] < 1)

def searchOptimalTriangle(start_point, paths):
    # We fix the closest point as one of the vertices of the triangle
    # and search for the other two iteratively
    path1 = paths[0, ...]
    paths = paths[1:, ...]
    #print(paths)
    triangle_paths = []
    while paths.shape[0] > 1:
        if isInTriangle(start_point, path1[0], paths[0, 0, :], paths[1, 0, :]):
            print("triangle at {}, {}, {}".format(path1[0], paths[0, 0, :], paths[1, 0, :]))
            triangle_paths = [path1, paths[0, ...], paths[1, ...]]
            return np.array(triangle_paths)
        paths = paths[1:, ...]
    return None


if __name__ == '__main__':
    filepath = "./paths.txt"
    x, y = readFile(filepath)
    #print("x = {}".format(x))
    start_points = [(0.8, 1.8), (2.2, 1.0), (2.7, 1.4)]
    for idx, start_point in enumerate(start_points):
        right, left = classifyPoints(x, y)
        if start_point[0] > start_point[1]:
            sorted_paths = sortPaths(start_point, right)
        else:
            sorted_paths = sortPaths(start_point, left)
        triangle_paths = searchOptimalTriangle(start_point, sorted_paths)

        if triangle_paths.any() == None:
            print("Triangle paths not found for the start point!")

        alpha_i, alpha_j, alpha_k = calcAlpha(start_point, triangle_paths[:, 0, :])
        newPath = getNewPath(triangle_paths[:, 1:, :], alpha_i, alpha_j, alpha_k)
        newPath = np.insert(newPath, 0, start_point, axis = 0)
        print(newPath)
    # interpolate path as cubic spline interpolation
        timesteps = np.arange(0, 50)
        fx = interp1d(timesteps, newPath[:, 0], kind='cubic')
        fy = interp1d(timesteps, newPath[:, 1], kind='cubic')
        plotPath(idx, triangle_paths, fx, fy)
    plt.show()

