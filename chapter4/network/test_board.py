import numpy as np
from chapter4.network.board import Board
from chapter4.network.board import recorder

def test_score_board():
    board = Board(dim=3)
    board.board = np.arange(1, 10).reshape(3,3)
    score = board.calculate_row_score(0)
    assert score == 6
    score = board.calculate_row_score(1)
    assert score == 15
    score = board.calculate_row_score(2)
    assert score == 24
    score = board.calculate_col_score(0)
    assert score == 12
    score = board.calculate_col_score(1)
    assert score == 15
    score = board.calculate_col_score(2)
    assert score == 18
    score = board.calculate_diag_score()
    assert score == 15
    score = board.calculate_diag_score(left=False)
    assert score == 15


def test_is_win():
    board = Board(dim=3)
    board.board = np.matrix([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])
    player = board.is_win()
    assert player == board.player_o
    board.board = np.matrix([
        [-1, 1, 0],
        [-1, 1, 0],
        [-1, 1, 0],
    ])
    player = board.is_win()
    assert player == board.player_x
    board.board = np.matrix([
        [0, 0, -1],
        [1, 1, 1],
        [0, -1, 0],
    ])
    player = board.is_win()
    assert player == board.player_o
    board.board = np.matrix([
        [0, 0, -1],
        [1, -1, 1],
        [-1, -1, 0],
    ])
    player = board.is_win()
    assert player == board.player_x
    board.board = np.matrix([
        [1, 0, -1],
        [1, 1, 0],
        [-1, -1, 1],
    ])
    player = board.is_win()
    assert player == board.player_o


def test_find_position():
    np.random.seed(30)
    board = Board(dim=3)
#    board.board = np.arange(1, 10,dtype=int).reshape(3, 3)
    board.board = np.matrix([
        [1, 0, -1],
        [1, 1, 0],
        [-1, -1, 1],
    ])
    pos = board.find_position()
    assert pos == (1,2)


def test_play():
    np.random.seed(30)
    board = Board(dim=3)
    board.board = np.matrix([
        [1, 0, -1],
        [1, 1, 0],
        [-1, -1, 1],
    ])
    pos = board.find_position()
    board.play(board.player_o, pos)
    assert np.array_equal(board.board, np.matrix([
        [1, 0, -1],
        [1, 1, 1],
        [-1, -1, 1],
    ]))


def test_record():
    board = Board()
    board.play(board.player_o, (1, 2))
    assert np.array_equal(recorder.pop(), np.matrix([
        [ 0.,  0.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  0.]]))

def test_is_cell_empty():
    board = Board()
    board.board = np.arange(1, 10).reshape(3, 3)
    res = board.is_cell_empty()
    assert res == False
    board.board = np.matrix([
        [1, 0, -1],
        [1, 1, 0],
        [-1, -1, 1],
    ])
    res = board.is_cell_empty()
    assert res == True


def test_play_full_game():
    board = Board()
    player = board.player_x
    while board.is_cell_empty():
        pos = board.find_position()
        board.play(player, pos)
        winner = board.is_win()
        if winner:
            print('The winner is: {}'.format(winner))
            for game in recorder:
                print("{}\n{}".format('-'*15,game))
            return
        player = board.player_o if player == board.player_x else board.player_x
    print('This is a draw')
    for game in recorder:
        print("{}\n{}".format('-' * 15, game))

