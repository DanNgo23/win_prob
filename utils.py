import numpy as np

def onehotCategorical(req, limit):
    arr = np.zeros((limit,))
    arr[req-1] = 1.
    return arr
