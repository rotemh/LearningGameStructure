import copy
import random

import game
from snake_action import *


class SnakeBoard(game.Board):
    RED = 'R'
    BLACK = 'B'
    WIDTH = 40#80
    HEIGHT = 25#50
    NUM_FOOD_ITEMS = 25

    def __init__(self, state=None, turn=None):
        if state == None:
            state = {}
            red_snake = (2, [(3,1),
                             (2,1),
                             (1,1)])
            black_snake = (4, [(SnakeBoard.WIDTH-4,SnakeBoard.HEIGHT-2),
                               (SnakeBoard.WIDTH-3,SnakeBoard.HEIGHT-2),
                               (SnakeBoard.WIDTH-2,SnakeBoard.HEIGHT-2)])
            food = set([(random.randint(0, SnakeBoard.WIDTH-1),
                         random.randint(0, SnakeBoard.HEIGHT-1)) for i in xrange(SnakeBoard.NUM_FOOD_ITEMS)])
            state['width'] = SnakeBoard.WIDTH
            state['height'] = SnakeBoard.HEIGHT
            state[SnakeBoard.RED] = red_snake
            state[SnakeBoard.BLACK] = black_snake
            state['food'] = food
            self.state = state
            self.turn = SnakeBoard.RED
        else:
            self.state = state
            self.turn = turn

    def get_legal_actions(self):
        actions = set()
        direction, _ = self.state[self.turn]
        
        for next_direction in [1,2,3,4]:
            if next_direction != direction and next_direction % 2 == direction % 2:
                continue # going backwards is not allowed
            action = SnakeAction(self.turn, next_direction)
            actions.add(action)

        return actions

    def _is_border_collision(self, coor):
        x, y = coor
        x_out = x < 0 or x >= SnakeBoard.WIDTH
        y_out = y < 0 or y >= SnakeBoard.HEIGHT

        return x_out or y_out

    def is_terminal(self):
        illegal_positions = set()
        _, red_snake = self.state[SnakeBoard.RED]
        _, black_snake = self.state[SnakeBoard.BLACK]

        if self.turn == SnakeBoard.RED:
            illegal_positions.update(red_snake)
            illegal_positions.update(black_snake[1:])
            coor = black_snake[0]
        else:
            illegal_positions.update(red_snake[1:])
            illegal_positions.update(black_snake)
            coor = red_snake[0]
        
        return coor in illegal_positions or self._is_border_collision(coor)

    def manhattan_dist(self, coor1, coor2):
        x1, y1 = coor1
        x2, y2 = coor2

        return abs(x1-x2) + abs(y1-y2)

    def closest_food_item_dist(self, color):
        _, snake = self.state[color]
        head = snake[0]

        return min([self.manhattan_dist(head, item) for item in self.state['food']])

    def reward_vector(self):
        end_game_val = 0.0
        if self.is_terminal():
            end_game_val = 2500.0
            if self.turn == SnakeBoard.BLACK:
                end_game_val *= -1

        length_scale_factor = 1000.0
        red_length = len(self.state[SnakeBoard.RED][1])
        black_length = len(self.state[SnakeBoard.BLACK][1])
        
        #return (red_length + end_game_val,
        #        black_length - end_game_val)
        
        return (red_length * length_scale_factor + end_game_val,
                black_length * length_scale_factor - end_game_val)

    def current_player_id(self):
        if self.turn == SnakeBoard.RED:
            return 0
        else:
            return 1


    def get_grid(self):
        grid = [[" " for j in xrange(self.state['width'])] for i in xrange(self.state['height'])]
        for item in self.state['food']:
            grid[item[1]][item[0]] = '@'
        
        _, red_snake = self.state[SnakeBoard.RED]
        _, black_snake = self.state[SnakeBoard.BLACK]
        
        for coor in red_snake:
            x, y = coor
            grid[y][x] = SnakeBoard.RED
        
        for coor in black_snake:
            x, y = coor
            grid[y][x] = SnakeBoard.BLACK

        return grid

    def visualize(self):
        _, red_snake = self.state[SnakeBoard.RED]
        _, black_snake = self.state[SnakeBoard.BLACK]

        grid = self.get_grid()
        grid[red_snake[0][1]][red_snake[0][0]] = '+'
        grid[black_snake[0][1]][black_snake[0][0]] = 'x'

        print ''.join(['_' for i in xrange(self.state['width'] + 2)])

        for i in xrange(self.state['height']):
            print '|' + ''.join(grid[self.state['height'] - i -1]) + '|'

        print ''.join(['-' for i in xrange(self.state['width'] + 2)])

        print
        
    def json_visualize(self):
        red_dir, red_snake = self.state[SnakeBoard.RED]
        black_dir, black_snake = self.state[SnakeBoard.BLACK]
        food = self.state['food']

        return {
            "redSnake": list(red_snake),
            "blackSnake": list(black_snake),
            "redDir": red_dir,
            "blackDir": black_dir,
            "food": list(food),
            "finished": self.is_terminal(),
            "player": self.current_player_id()
        }

    def __copy__(self):
        new_state = copy.deepcopy(self.state)
        
        return SnakeBoard(new_state, self.turn)

    def get_snake(self, color):
        return self.state[color]

    def set_snake(self, color, direction, coors):
        self.state[color] = (direction, coors)

    def has_food(self, coor):
        return coor in self.state['food']

    def get_food(self):
        return self.state['food']

    def switch_turn(self, color):
        if color == SnakeBoard.RED:
            self.turn = SnakeBoard.BLACK
        else:
            self.turn = SnakeBoard.RED
