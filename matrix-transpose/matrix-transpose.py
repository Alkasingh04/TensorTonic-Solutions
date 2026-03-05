import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    matrix = np.array(A)
    return matrix.T


A = [[1,2,3],
     [4,5,6]]

result = matrix_transpose(A)
print(result)