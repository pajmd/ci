import numpy as np


how = np.matrix([[1,2], [3,4], [5,6]])
do = np.matrix([[2,4]])
do_on_how = do / how
print(do_on_how)
how = how + do_on_how
print(do_on_how)


x1 = np.arange(start=1, stop=7.0).reshape((3, 2))
x1.T
x1c1 = x1[:, 0]
x2 = np.arange(start=1, stop=4.0)
x3 = x1c1 * x2
print(x3)
x1 = np.arange(3.0).reshape((1, 3))
x2 = np.arange(3.0)
x3 = np.multiply(x1, x2)
print(x3)

x1 = np.arange(3.0).reshape((3, 1))
x2 = np.arange(start=1, stop=4.0)
x3 = x2 *x1
print(x3)

x1 = np.array([1,2,3])
x2 = np.array([3,4,5])
x3 = x1 * x2
print(x3)

m = np.matrix([[-0.52886962],
 [-0.26375519],
 [-0.14401064]])
mc = m.copy()
pass