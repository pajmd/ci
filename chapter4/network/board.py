import numpy as np
from collections import deque
from collections import namedtuple
from functools import wraps

class Match(object):
    def __init__(self):
        self.games = deque()
        self.turns = 0
        self.winner = 0

Recorder = namedtuple('Recorder','games winners')
recorder = Recorder(deque(), deque())

class Recorder(object):
    def __init__(self):
        self.matches = deque()
        match = Match()
        self.match = match
        self.matches.append(match)

    def set_winner(self, winner):
        self.match.winner = winner

    def inc_turns(self):
        self.match.turns += 1

    def save_game(self, game):
        self.match.games.append(game)

    def new_match(self):
        match = Match()
        self.match = match
        self.matches.append(match)

recorder = Recorder()

def record_game(recorder):
    def top_wraper(func_game):
        #@wraps(func_game)
        def wrapper(*arg, **kwarg):
            ret = func_game(*arg, **kwarg)
#            recorder.games.append(arg[0].board.copy())
            recorder.save_game(arg[0].board.copy())
            recorder.inc_turns()
            return ret
        return wrapper
    return top_wraper

def record_winner(recorder):
    def top_wraper(func_game):
        #@wraps(func_game)
        def wrapper(*arg, **kwarg):
            ret = func_game(*arg, **kwarg)
#            recorder.winners.append(ret)
            recorder.set_winner(ret)
            return ret
        return wrapper
    return top_wraper

def record_new_match(recorder):
    def top_wraper(func_game):
        #@wraps(func_game)
        def wrapper(*arg, **kwarg):
            ret = func_game(*arg, **kwarg)
            recorder.new_match()
            return ret
        return wrapper
    return top_wraper

class Board(object):

    def __init__(self, dim=3):
        self.board = np.zeros((dim,dim),dtype=float)
        self.player_x = -1
        self.player_o = 1
        self.dim = dim
        self._draw = 0
        self.turns = 0

    @record_new_match(recorder)
    def reset(self):
        self.board = np.zeros((self.dim, self.dim), dtype=float)
        self.turns = 0

    def calculate_row_score(self, row_index):
        return np.sum(self.board[row_index])

    def calculate_col_score(self, col_index):
        return np.sum(self.board.T[col_index])

    def calculate_diag_score(self,left=True):
        if left:
            return self.board.trace()
        return np.fliplr(self.board).trace()

    def winner_is(self, score):
        if score == -self.dim:
            return self.player_x
        if score == self.dim:
            return self.player_o
        return 0

    def is_win(self):
        for i in range(self.dim):
            player = self.winner_is(self.calculate_row_score(i))
            if player:
                return player
        for i in range(self.dim):
            player = self.winner_is(self.calculate_col_score(i))
            if player:
                return player
        player = self.winner_is(self.calculate_diag_score())
        if player:
            return player
        return self.winner_is(self.calculate_diag_score(left=False))

    def is_cell_empty(self):
        return 0 in self.board

    def find_position(self):
        row_col = tuple(np.random.random_integers(0, self.dim -1, 2))
        while self.board.item(row_col):
            row_col = tuple(np.random.random_integers(0, self.dim - 1, 2))
        return row_col

    @record_game(recorder)
    def play(self, player, position):
        self.board[position] = player
        #return self.board

    def set_the_game(self):
        pass

    @record_winner(recorder)
    def play_full_game(self, player):
        while board.is_cell_empty():
            pos = board.find_position()
            board.play(player, pos)
            winner = board.is_win()
            if winner:
                # print('The winner is: {}'.format(winner))
                # for game in recorder.games:
                #     print("{}\n{}".format('-' * 15, game))
                return winner
            player = board.player_o if player == board.player_x else board.player_x
        # print('This is a draw')
        # for game in recorder.games:
        #     print("{}\n{}".format('-' * 15, game))
        return self._draw


if __name__ == '__main__':
    board = Board()
    # # rec = record_game(recorder)
    # # play = rec(board.play)
    # board.play(board.player_o, (1, 2))
    # print(recorder)
    np.random.seed(4)
    player = board.player_x
    for i in range(100000):
        board.play_full_game(player)
        board.reset()
    count_palyer_o_winner = 0
    for i, match in enumerate(recorder.matches):
        if match.winner == board.player_o and match.turns < 6:
            count_palyer_o_winner += 1
            print('Game {}\n{}'.format(i,'*'*10))
            print('The winner is: {} no turns = {}'.format('X' if match.winner == board.player_x
                                                           else 'O' if match.winner == board.player_o
                                                            else 'Drawwwwwwwwwwwwwwwwwww',
                                                           match.turns))
            for game in match.games:
                print("{}\n{}".format('-' * 15, game))
    print('Number of games won by O: %d'%count_palyer_o_winner)
