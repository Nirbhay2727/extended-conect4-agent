from json.encoder import INFINITY
import random
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer, get_diagonals_primary, get_diagonals_secondary
import sys

# Connect 4 AI

alpha = -np.inf
beta =  np.inf

def update_board(state, column: int, player_num: int, is_popout: bool = False):
        board, num_popouts = state
        if not is_popout:
            if 0 in board[:, column]:
                for row in range(1, board.shape[0]):
                    update_row = -1
                    if board[row, column] > 0 and board[row - 1, column] == 0:
                        update_row = row - 1
                    elif row == board.shape[0] - 1 and board[row, column] == 0:
                        update_row = row
                    if update_row >= 0:
                        board[update_row, column] = player_num
                        break
            else:
                err = 'Invalid move by player {}. Column {}'.format(player_num, column, is_popout)
                raise Exception(err)
        else:
            if 1 in board[:, column] or 2 in board[:, column]:
                for r in range(board.shape[0] - 1, 0, -1):
                    board[r, column] = board[r - 1, column]
                board[0, column] = 0
            else:
                err = 'Invalid move by player {}. Column {}'.format(player_num, column)
                raise Exception(err)
            num_popouts[player_num].decrement()

class AIPlayer:
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        # Do the rest of your implementation here

    def evaluationFunction(self,playerNumber: int, state: Tuple[np.array, Dict[int, Integer]]) -> int:
        """
        Given the current state of the board, return the evaluation score
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: evaluation score
        """
        # Do the rest of your implementation here

        score = get_pts(playerNumber, state[0])-get_pts(3-playerNumber, state[0])*state[0].shape[1]*state[0].shape[0]//4
        # Prefer having your pieces in the center of the board.
        # score boost
        numrow,numcol = state[0].shape
        for row in range(numrow):
            for col in range(numcol):
                if state[0][row][col] == playerNumber:
                    score -= abs(numcol//2-col)*abs(numrow//2-row)
                elif state[0][row][col] == 3-playerNumber:
                    score += abs(numcol//2-col)*abs(numrow//2-row)

        # score = get_pts(3-playerNumber, state[0])*state[0].shape[1]*state[0].shape[0]//15
        # # Prefer having your pieces in the center of the board.
        # # score boost
        # numrow,numcol = state[0].shape
        # for row in range(numrow):
        #     for col in range(numcol):
        #         if state[0][row][col] == playerNumber:
        #             score -= abs(numcol//2-col)
        #         elif state[0][row][col] == 3-playerNumber:
        #             score += abs(numcol//2-col)


        return score
        raise NotImplementedError('Whoops I don\'t know what to do')

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move
        This will play against either itself or a human player
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here

        CUTOFF = 4

        #minimax with alpha beta pruning

        def max_value(state: Tuple[np.array, Dict[int, Integer]], depth: int) -> int:
            global alpha,beta
            if get_valid_actions(self.player_number, state) == [] or len(get_valid_actions(self.player_number, state)) == 0:
                return get_pts(self.player_number, state[0])
            if depth == CUTOFF:
                return self.evaluationFunction(self.player_number,state)
            v = -np.inf
            for action in get_valid_actions(self.player_number,state):
                v = max(v, min_value(result(state, action,self.player_number), depth+1))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return (v)

        def min_value(state: Tuple[np.array, Dict[int, Integer]], depth: int) -> int:
            global alpha,beta
            if get_valid_actions(3-self.player_number, state) == [] or len(get_valid_actions(self.player_number, state)) == 0:
                return get_pts(3-self.player_number, state[0])
            if depth == CUTOFF:
                return self.evaluationFunction(3-self.player_number,state)
            v = np.inf
            for action in get_valid_actions(3-self.player_number,state):
                v = min(v, max_value(result(state, action,3-self.player_number), depth+1))
                beta = min(beta,  v)
                if(beta<=alpha):
                    break
            return v

        def result(state: Tuple[np.array, Dict[int, Integer]], action: Tuple[int, bool], playerNumber:int) -> Tuple[np.array, Dict[int, Integer]]:
            # print("action",action)
            new_board = np.copy(state[0])
            new_dict = state[1].copy()
            column, is_popout = action
            update_board((new_board, new_dict), column, playerNumber, is_popout)
            return (new_board, new_dict)

        def minimax_decision(state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
            global alpha,beta
            best_score = -np.inf
            best_action = None
            # print (state)
            for action in get_valid_actions(self.player_number,state):
                new_state = result(state, action, self.player_number)
                # print(new_state)
                v = max_value(new_state, 0)
                if v > best_score:
                    best_score = v
                    best_action = action
            print(alpha," ",beta)
            return best_action
        # print(state)
        return minimax_decision(state)

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here

        CUTOFF = 2

        #expectimax

        def max_value(state: Tuple[np.array, Dict[int, Integer]], depth: int) -> int:
            if get_valid_actions(self.player_number, state) == [] or len(get_valid_actions(self.player_number, state)) == 0:
                return get_pts(self.player_number, state[0])
            if depth == CUTOFF:
                return self.evaluationFunction(self.player_number,state)
            v = -np.inf
            for action in get_valid_actions(self.player_number,state):
                v = max(v, exp_value(result(state, action,self.player_number), depth+1))
            return v

        def exp_value(state: Tuple[np.array, Dict[int, Integer]], depth: int) -> int:
            if get_valid_actions(3-self.player_number, state) == [] or len(get_valid_actions(self.player_number, state)) == 0:
                return get_pts(self.player_number, state[0])
            if depth == CUTOFF:
                return self.evaluationFunction(3-self.player_number,state)
            v = 0
            l = len(get_valid_actions(3-self.player_number,state))
            for action in get_valid_actions(3-self.player_number,state):
                v += max_value(result(state, action,3-self.player_number), depth+1)
            return v/l

        def result(state: Tuple[np.array, Dict[int, Integer]], action: Tuple[int, bool], playerNumber:int) -> Tuple[np.array, Dict[int, Integer]]:
            print("action",action)
            new_board = np.copy(state[0])
            new_dict = state[1].copy()
            column, is_popout = action
            update_board((new_board, new_dict), column, playerNumber, is_popout)
            return (new_board, new_dict)

        def expectimax_decision(state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
            best_score = -np.inf
            best_action = None
            # print (state)
            for action in get_valid_actions(self.player_number,state):
                new_state = result(state, action, self.player_number)
                # print(new_state)
                v = max_value(new_state, 0)
                if v > best_score:
                    best_score = v
                    best_action = action
            return best_action
        
        return expectimax_decision(state)