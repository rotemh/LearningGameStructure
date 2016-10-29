import copy
import random

import game


class SnakeAction(game.Action):
    """
    This board represents an action in Snake.
    The actions specifies the color of the snake
    and its direction.
    Direction is defined as follow:
    1 - Left
    2 - Up
    3 - Right
    4 - Down
    """

    def __init__(self, color, direction):
        """
        params:
        color - a string from ['R', 'B'] that represents the color of the snake
        direction - an integer from [1,2,3,4] representing the direction to move the snake

        Functions needed:
        board.getSnakeCoords(color)
        board.setSnakeCoords(color, coordinates)
        board.hasFood(coordinate)
        """

        self.color = color
        self.direction = direction

    def apply(self, board):
        if not board.is_legal_action(self):
            raise Exception('This action is not allowed! => {}'.format(self))

        # copy
        new_board = copy.copy(board)

        # Grab the coordinates of the snake segments
        _, old_snake_coords = new_board.get_snake(self.color)
        old_snake_head = old_snake_coords[0]

        # Calculate position of the snake head
        if self.direction == 1: # move up
            new_snake_head = (old_snake_head[0], old_snake_head[1] + 1)
        elif self.direction == 2:# move right
            new_snake_head = (old_snake_head[0] + 1, old_snake_head[1])
        elif self.direction == 3: # move down
            new_snake_head = (old_snake_head[0], old_snake_head[1] - 1)
        else: # move left
            new_snake_head = (old_snake_head[0] - 1, old_snake_head[1])

        # Create new snake
        new_snake_coords = [new_snake_head]
        new_snake_coords.extend(old_snake_coords[:-1])

        # If the snake ate food, then we don't drop the final segment of the snake
        if new_board.has_food(new_snake_head):
            # keep the end of the snake
            new_snake_coords.append(old_snake_coords[-1])

            # replace the food item
            food = new_board.get_food()
            food.discard(new_snake_head)
            new_food_item = (random.randint(0, new_board.state['width']-1),
                             random.randint(0, new_board.state['height']-1))

            food.add(new_food_item)

        # update board with new snake coordinates and direction
        new_board.set_snake(self.color, self.direction, new_snake_coords)

        # change the turn
        new_board.switch_turn(self.color)

        return new_board

    def __hash__(self):
        return hash((self.color, self.direction))

    def __repr__(self):
        return 'SnakeAction(color={},direction={})'.format(self.color,self.direction)
