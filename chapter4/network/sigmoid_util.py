import numpy as np

def s(x):
    return 1/(1+np.exp(-x))

def ds(x):
    return s(x)*(1-s(x))

