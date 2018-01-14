from chapter4.network.n_network import Network
import numpy as np

def _test_forward():
    mynet = Network(2,1,3)
    mynet.create_connections()
    mynet.feed_forward([1,1])
    print(mynet.output_ouputs)

def _test_ouput_deltas():
    mynet = Network(4,2,3)
    mynet.create_connections()
    mynet.feed_forward([1,1,1,1])
    print(mynet.output_ouputs)
    mynet.matrix_who = np.matrix([[0.11,0.12,0.13],[0.21,0.22,0.23]])
    mynet.output_ouputs = np.matrix([[0.1,0.2,0.3]])
    target = [0.9, 0.8, 0.7]
    ods = mynet.get_output_deltas(target)
    np.testing.assert_array_almost_equal(np.matrix([
        (0.9-0.1)*mynet.gradient(0.1),
        (0.8 - 0.2) * mynet.gradient(0.2),
        (0.7 - 0.3) * mynet.gradient(0.3)])
    , ods)

def _test_hidden_deltas():
    mynet = Network(4,2,3)
    mynet.create_connections()
    mynet.feed_forward([1,1,1,1])
    print(mynet.output_ouputs)
    mynet.matrix_who = np.matrix([[0.11,0.12,0.13],[0.21,0.22,0.23]])
    mynet.output_ouputs = np.matrix([[0.1,0.2,0.3]])
    target = [0.9, 0.8, 0.7]
    ods = mynet.get_output_deltas(target)
    mynet.matrix_who = np.matrix([[1, 2, 3], [4, 5, 6]])
    mynet.hidden_outputs = np.matrix([[0.5, 0.6]])
    ods = np.matrix([[0.05, 0.06, 0.07]])
    hds = mynet.get_hidden_deltas(ods)

    np.testing.assert_array_almost_equal(np.matrix([mynet.gradient(0.5)*0.38, mynet.gradient(0.6) * 0.92])
    , hds)


def _test_back_propagate():
    mynet = Network(4,2,3)
    mynet.create_connections()
    mynet.feed_forward([1,1,1,1])
    print(mynet.output_ouputs)
    mynet.matrix_who = np.matrix([[0.11,0.12,0.13],[0.21,0.22,0.23]])
    mynet.output_ouputs = np.matrix([[0.1,0.2,0.3]])
    target = [0.9, 0.8, 0.7]
    mynet.back_propagate(target)

def test_train_xor():
    mynet = Network(2,3,2)
    mynet.create_connections()
    # print(mynet.matrix_wih)
    # print(mynet.matrix_who)
    for i in range(1000):
        # 1 xor 1 = 0 [prob 0, prob1]
        mynet.train([1,1],[0], learning_rate=1)
        # print(mynet.matrix_wih)
        # print(mynet.matrix_who)https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        # mynet.train([1, 0], [0,1], learning_rate=1)
        # mynet.train([0, 0], [1,0], learning_rate=1)
        # mynet.train([0, 1], [0, 1], learning_rate=1)
    mynet.feed_forward([1,1])
    print('[1,1]= %s'%mynet.output_ouputs)
    # mynet.feed_forward([0,0])
    # print('[0,0]= %s'%mynet.output_ouputs)
    # mynet.feed_forward([1,0])
    # print('[1,0]= %s'%mynet.output_ouputs)
    # mynet.feed_forward([0,1])
    # print('[0,1]= %s'%mynet.output_ouputs)
    # mynet.feed_forward([1,1])
    # print('[1,1]= %s'%mynet.output_ouputs)

