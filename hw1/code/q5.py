import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def estimateRotationTranslation(P, Q):
    p_centroid = np.array(np.sum(P, axis=0)/len(P))
    q_centroid = np.array(np.sum(Q, axis=0)/len(Q))

    P_normalized = P - p_centroid
    Q_normalized = Q - q_centroid

    # Compute the covariance matrix H
    H = np.matmul(P_normalized.T, Q_normalized)

    # obtain the SVD decomposition of the covariance matrix
    U, s, Vt = linalg.svd(H)
    V = Vt.T
    Ut = U.T
    estimated_R = np.dot(V, Ut)

    # eliminate chance of reflection
    if round(linalg.det(estimated_R)) == -1:
        estimated_R[3] *= -1

    estimated_t = -np.matmul(estimated_R, p_centroid) + q_centroid

    return estimated_R, estimated_t

if __name__=='__main__':
    # Generate 50 random point in 3d space
    x = np.random.uniform(0, 2, 50)
    y = np.random.uniform(0, 2, 50)
    z = np.random.uniform(0, 2, 50)
    P = np.array(list(zip(x, y, z))).T

    # Generate prior R and t matrix to transform P to Q
    R = linalg.orth(np.random.rand(3, 3))
    if round(linalg.det(R)) == -1:
        R[3] *= -1
    t = np.random.rand(3, 1)

    # Generate Q points from assumed R and t
    Q = np.matmul(R, P)
    Q = Q + t

    # obtain estimated R and t according to proposed algorithm
    estimated_R, estimated_t = estimateRotationTranslation(P.T, Q.T)
    print("Original R = {}, Estimated R = {} ".format(R, estimated_R))
    print("Original t = {}, Estimated t = {}".format(t, estimated_t))
    print("Error in Rotation matrix = {}".format(np.abs(R - estimated_R)))
    print("Error in translation vector elements = {}".format(np.abs(t - estimated_t)))

    # generated Q from estimated_R and estimated_t
    estimated_Qt = np.matmul(estimated_R, P).T + estimated_t
    estimated_Q = estimated_Qt.T

    # Plot P and original Q
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(P[0, :], P[1, :], P[2, :], c='green', marker='o', label='P')
    ax.scatter(Q[0, :], Q[1, :], Q[2, :], c='red', marker='x', label='Q')
    ax.set_xlabel('X'); ax.set_ylabel('Y'), ax.set_zlabel('Z')
    plt.legend(loc='lower left')
    plt.title("3d points P and Q = transformed by original R and t")

    # Plot P and estimated Q
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(P[0, :], P[1, :], P[2, :], c='green', marker='o', label='P')
    ax.scatter(estimated_Q[0, :], estimated_Q[1, :], estimated_Q[2, :], c='red', marker='x', label='Q')
    ax.set_xlabel('X'); ax.set_ylabel('Y'), ax.set_zlabel('Z')
    plt.legend(loc='lower left')
    plt.title("3d points P and Q transformed by estimated R and t")
    plt.show()

