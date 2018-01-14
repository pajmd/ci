from chapter4.network.n_network import Network
import numpy as np


def character(glyph):
    return list(map(lambda c: 1 if c == '#' else 0, glyph))

def alphabet():
    a = character(
        '.#####.' +
        '#.....#' +
        '#.....#' +
        '#######' +
        '#.....#' +
        '#.....#' +
        '#.....#'
    )

    b = character(
        '######.' +
        '#.....#' +
        '#.....#' +
        '######.' +
        '#.....#' +
        '#.....#' +
        '######.'
    )

    c = character(
        '#######' +
        '#......' +
        '#......' +
        '#......' +
        '#......' +
        '#......' +
        '#######'
    )

    altered_c = character(
      '#######' +
      '#......' +
      '#......' +
      '#......' +
      '#......' +
      '###....' +
      '#######'
    )
    return (a, [0.1]), (b, [0.3]), (c, [0.5]), altered_c

def test_train_ocr():
    a, b ,c, c_to_recognized = alphabet()
    num_input_neurons = len(a[0])
    mynet = Network(num_input_neurons, 3 * num_input_neurons, 1)
    mynet.create_connections()
    # print(mynet.matrix_wih)
    # print(mynet.matrix_who)
    repeats = 100
    for i in range(repeats):
        mynet.train(a[0],a[1], learning_rate=1)
    for i in range(repeats):
        mynet.train(b[0],b[1], learning_rate=1)
    for i in range(repeats):
        mynet.train(c[0],c[1], learning_rate=1)
        # mynet.train([1, 0], [0,1], learning_rate=1)
        # mynet.train([0, 0], [1,0], learning_rate=1)
    mynet.feed_forward(c_to_recognized)
    print('c_to_recognized= %s'%mynet.output_ouputs)

