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
def _test_train_urls():
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
    print('Results:')
    mynet.feed_forward([1,0,1])
    print('{} expected {}'.format(mynet.output_ouputs,[1,0,0]))
    mynet.feed_forward([0,1,1])
    print('{} expected {}'.format(mynet.output_ouputs,[0, 1, 0]))
    mynet.feed_forward([1,0,1])
    print('{} expected {}'.format(mynet.output_ouputs,[1,0,0]))
    mynet.feed_forward([0,0,1])
    print('{} expected {}'.format(mynet.output_ouputs,[1,0,0]))

    # target = [1, 0, 0]
    # mynet.back_propagate(target)
    # mynet.feed_forward([1,1,0])
    # print(mynet.output_ouputs)


def test_train_xor():
    mynet = Network(2,3,2)
    mynet.create_connections()
    # print(mynet.matrix_wih)
    # print(mynet.matrix_who)
    for i in range(1000):
        # 1 xor 1 = 0 [prob 0, prob1]
        mynet.train([1,1],[1,0], learning_rate=1)
        # print(mynet.matrix_wih)
        # print(mynet.matrix_who)https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        mynet.train([1, 0], [0,1], learning_rate=1)
        mynet.train([0, 0], [1,0], learning_rate=1)
        mynet.train([0, 1], [0, 1], learning_rate=1)
    mynet.feed_forward([1,1])
    print('[1,1]= %s'%mynet.output_ouputs)
    mynet.feed_forward([0,0])
    print('[0,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([1,0])
    print('[1,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([0,1])
    print('[0,1]= %s'%mynet.output_ouputs)
    mynet.feed_forward([1,1])
    print('[1,1]= %s'%mynet.output_ouputs)


def test_train_xor_1o():
    mynet = Network(2,3,1)
    mynet.create_connections()
    # print(mynet.matrix_wih)
    # print(mynet.matrix_who)
    for i in range(1000):
        # 1 xor 1 = 0 [prob 0, prob1]
        mynet.train([1,1],[0], learning_rate=1)
        # print(mynet.matrix_wih)
        # print(mynet.matrix_who)https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
        mynet.train([1, 0], [1], learning_rate=1)
        mynet.train([0, 0], [0], learning_rate=1)
        # mynet.train([0, 1], [1], learning_rate=1)
    mynet.feed_forward([1,1])
    print('[1,1]= %s'%mynet.output_ouputs)
    mynet.feed_forward([0,0])
    print('[0,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([1,0])
    print('[1,0]= %s'%mynet.output_ouputs)
    mynet.feed_forward([0,1])
    print('[0,1]= %s'%mynet.output_ouputs)
    mynet.feed_forward([1,1])
    print('[1,1]= %s'%mynet.output_ouputs)


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

def _test_update_who():
    mynet = Network(3,4,3)
    mynet.create_connections()

    mynet.matrix_who = np.matrix(
        [[-0.5910955,   0.75623487, -0.94522481],
        [0.34093502, -0.1653904,   0.11737966],
        [-0.71922612, -0.60379702, 0.60148914],
        [0.93652315, -0.37315164,  0.38464523]])

    d = [0.40795614,0.62674606,0.23841622,0.49377636]
    d = [1, 2, 3]
    mynet.hidden_outputs = np.array(d, dtype=float)
    output_deltas = np.array([ 0.10676222, -0.11685494, -0.12631629], dtype=float)
    # c = output_deltas * h
    #result is addition of c to each row of matric_who
    matrix_who_expected = np.matrix(
        [[-0.56931835,  0.73239903, -0.97099057],
         [0.37439142, -0.20200958,  0.07779554],
        [-0.7064992, -0.61772708, 0.58643121],
        [0.96288148, -0.40200175,  0.35345923]]
    )
    mynet.update_hidden_output_weight_new(output_deltas, 0.5)
    assert mynet.matrix_who == matrix_who_expected

def _test_brodacasting_add():
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)+1
    # broadcasting happens meaning x 2 is extending to the smae shape as x1 and the x2 value repeated for each row.
    r = np.add(x1, x2)
    print(x1)
    print(x2)
    print(r)


def _test_brodacasting_plus():
    x1 = np.arange(9.0).reshape((3, 3))
    x2 = np.arange(3.0)+1
    r = x1 + x2
    print(x1)
    print(x2)
    print(r)