import numpy as np
from chapter4.network.sigmoid_util import s, ds

def test_sigmoid_1_235():
    print('sig(1.235) = {}'.format(s(1.235)))
    derivative = ds(1.235)
    print('dsig(1.235) = {}'.format(derivative))
    error = 0 - s(1.235)
    print('dsig(1.235) * error= {}'.format(derivative * error))
    assert derivative == s(1.235)*(1-s(1.235))
    dw = np.array([-0.1838, -0.1710, -0.1920])
    oldw = np.array([0.3, 0.5, 0.9])
    res = oldw + dw
    print(res)
    #assert res == np.array([0.1162, 0.329, 0.708])



def test_derivative_sigmoid_1_235():
    sr= s(1.235)
    print('sig(1.235) = {}'.format(sr))
    derivative = ds(sr)
    print('dsig({}) = {}'.format(sr, derivative))
    assert derivative == s(1.235)*(1-s(1.235))

def test_cal_new_w():
    derivative = ds(1.235)
    error = 0 - s(1.235)
    douput = derivative * error
    dw = douput * np.array([0.73105, 0.78583, 0.69997])
    oldw = np.array([0.3, 0.5, 0.9])
    res = oldw + dw
    print('new weight: {}'.format(res))
    print(0.20114892 * 0.73 + 0.39374168 * 0.79 + 0.80535149 * 0.69)

def test_neww():
    o = 0.1162 * 0.73 + 0.329 * 0.79 + 0.708 * 0.69
    print('output={} sig(o)= {}'.format(o, s(o)))