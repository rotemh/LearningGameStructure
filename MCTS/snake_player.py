import game
from snake_action import *


class SnakePlayer(game.HumanPlayer):
    """
    Human player that plays Snake.
    """

    def choose_action(self, board):
        next_dir = self.source() # returns 0,1,2,3,4, representing the direction the person wants to go
        color = board.turn
        curr_dir = board.get_snake(color)[0] # current direction the snake is going
        
        # if human isn't holidng the joystick in any direction
        if next_dir == 0:
            action = SnakeAction(board.turn, curr_dir)
        elif (next_dir == 1 and curr_dir == 3) or (next_dir == 3 and curr_dir == 1):
            action = SnakeAction(board.turn, curr_dir)
        elif (next_dir == 2 and curr_dir == 4) or (next_dir == 4 and curr_dir == 2):
            action = SnakeAction(board.turn, curr_dir)
        else:
            action = SnakeAction(board.turn, next_dir)

        return action
