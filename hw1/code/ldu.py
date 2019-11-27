import numpy as np
import pprint

def permMatrix(matrix):
    """ Returns a permutation matrix for LDU decomposition """
    length = matrix.shape[0]
    identity = np.identity(length)

    """ Rearrange the identity matrix such that max element of row 
        is in the diagonal of the matrix.
    """
    for j in range(length):
        row_idx = max(range(j, length), key = lambda i: abs(matrix[i][j])) 
        if j != row_idx:
            identity[row_idx], identity[j] = identity[j], identity[row_idx]
    return identity

def lduDecomposition(A):
    """ Implements LDU decomposition for a given matrix A 
        A must be an invertible square matrix.

        Returns: 
            P, L, D and U matrices
    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError(" A is not a square matrix!")

    n = len(A)

    D = U = np.zeros((n, n), dtype=float)
    L = np.identity(n)

    P = permMatrix(A)
    PA = np.matmul(P, A)
    
    for j in range(n):

        for i in range(j+1):
            s1 = sum(U[k][j]*L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j]*L[i][k] for k in range(i))
            L[i][j] = (PA[i][j] - s2)/U[j][j]

    return (P, L, U)


#A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]], dtype=float)
#A = np.array([[1, -2, 1], [1, 2, 2], [2, 3, 4]], dtype=float)
#A = np.array([[1, 1, 0], [1, 1, 2], [4, 2, 3]], dtype=float)
A = np.array([[10, 9, 2], [5, 3, 1], [2, 2, 2]], dtype=float)
#A = np.array([[16, 16, 0, 0], [4, 0, -2, 0], [0, 1, -1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=float)
#A = np.array([[10, 6, 4], [5, 3, 2], [1, 1, 0]], dtype=float)

P, L, U = lduDecomposition(A)
print("P = "); pprint.pprint(P)
print("L = "); pprint.pprint(L)
print("U = "); pprint.pprint(U)

# verify the solutions
print(" PA = ")
pprint.pprint(np.matmul(P, A))
print("LU = ")
pprint.pprint(np.matmul(L, U))
