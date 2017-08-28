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

def _test_result_from_taring():
    mynet = Network(3,2,3)
    mynet.create_connections()
    mynet.feed_forward([1,1,1])
    print(mynet.output_ouputs)
    target = [1, 0, 0]
    mynet.back_propagate(target)
    mynet.feed_forward([1,1,0])
    print(mynet.output_ouputs)

# >> wWorld,wRiver,wBank =101,102,103
# >> uWorldBank,uRiver,uEarth =201,202,203
def test_train():
    mynet = Network(3,4,3)
    mynet.create_connections()
    #mynet.feed_forward([1,1,1])
    #print(mynet.output_ouputs)
    print(mynet.matrix_wih)
    print(mynet.matrix_who)
    for i in range(1000):
        #print('='*20)
        mynet.train([1,0,1],[1,0,0])
        #print(mynet.matrix_wih)
        #print(mynet.matrix_who)
    # print('='*30)
    # for i in range(30):
        mynet.train([0, 1, 1], [0, 1, 0])
    # print('='*30)
    # for i in range(30):
        mynet.train([1, 0, 0], [0, 0, 1])

    mynet.feed_forward([1,0,1])
    print(mynet.output_ouputs)
    mynet.feed_forward([0,1,1])
    print(mynet.output_ouputs)
    mynet.feed_forward([1,0,1])
    print(mynet.output_ouputs)

    # target = [1, 0, 0]
    # mynet.back_propagate(target)
    # mynet.feed_forward([1,1,0])
    # print(mynet.output_ouputs)

def _test_train_xor():
    mynet = Network(2,3,1)
    mynet.create_connections()
    print(mynet.matrix_wih)
    print(mynet.matrix_who)
    for i in range(1):
        mynet.train([11,11],[0], learning_rate=1)
        print(mynet.matrix_wih)
        print(mynet.matrix_who)
        mynet.train([10, 11], [1], learning_rate=1)
        mynet.train([10, 11], [1], learning_rate=1)
        mynet.train([10, 10], [0], learning_rate=1)
    mynet.feed_forward([1,1])
    print('[1,1]= %s'%mynet.output_ouputs)
    mynet.feed_forward([10,10])
    print('[0,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([11,10])
    print('[1,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([10,11])
    print('[0,1]= %s'%mynet.output_ouputs)


def _test_train_31():
    mynet = Network(3,3,1)
    mynet.create_connections()
    print(mynet.matrix_wih)
    print(mynet.matrix_who)
    for i in range(10000):
        mynet.train([0,0,1],[0], learning_rate=1)
        # print(mynet.matrix_wih)
        # print(mynet.matrix_who)
        mynet.train([1,1,1], [1], learning_rate=1)
        mynet.train([1,0,1], [1], learning_rate=1)
        mynet.train([0, 1, 1], [0], learning_rate=1)
    mynet.feed_forward([1,0,0])
    print('[1,0,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([1,0,0])
    print('[0,0,0]= %s'%mynet.output_ouputs)
