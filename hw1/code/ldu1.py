import numpy as np
import pprint

def lduDecomposition(A):
    """ Returns the P, L, D, U, decomposition of a matrix
        given that A is square and invertible with partial pivoting
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(" A is not a square matrix")
    n = len(A)
    L = np.identity(n)
    U = A.copy()   # initially assume U = A for invariant LA' = A 
    P = np.identity(n)
    D = np.identity(n) 

    for j in range(n):
        for k in range(j+1, n):
            if U[k][j] != 0:
                #print(U[k], A[j])

                # swap the rows if any of the diagonal elements are 0
                # swap the lower triangular part of L matrix only
                if U[j][j] == 0.0:
                    U[[k, j]] = U[[j, k]]
                    P[[k, j]] = P[[j, k]]
                    L[[k, j], 0:j] = L[[j, k], 0:j]
                # Compute the U and L matrix according to the formula
                L[k][j] = U[k][j]/U[j][j]
                U[k] = U[k] + U[j]*(-U[k][j]/U[j][j])
    
    # Calculate the diagonal matrix from composite U matrix
    # if any of the diagonal elements are 0, special handling is required
    for j in range(n):
        if U[j][j] != 0:
            D[j][j] = U[j][j]
            U[j] /= U[j][j]
        else:
            D[j][j] = 0.0
            U[j][j] = 1.0
    return(P, L, D, U)



#A = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]], dtype=float)
#A = np.array([[1, -2, 1], [1, 2, 2], [2, 3, 4]], dtype=float)
#A = np.array([[1, 1, 0], [1, 1, 2], [4, 2, 3]], dtype=float)
A = np.array([[10, 9, 2], [5, 3, 1], [2, 2, 2]], dtype=float)
#A = np.array([[16, 16, 0, 0], [4, 0, -2, 0], [0, 1, -1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=float)
#A = np.array([[10, 6, 4], [5, 3, 2], [1, 1, 0]], dtype=float)

P, L, D, U = lduDecomposition(A)

print("P = ")
pprint.pprint(P)
print("L = ")
pprint.pprint(L)
print("D = ")
pprint.pprint(D)
print("U = ")
pprint.pprint(U)

# verify the solution through checking whether PA  = LDU

X = np.matmul(D, U)
print("composite U = ")
pprint.pprint(X)
RHS = np.matmul(L, X)
LHS = np.matmul(P, A)

print("PA = ")
pprint.pprint(LHS)
print("LDU = ")
pprint.pprint(RHS)
