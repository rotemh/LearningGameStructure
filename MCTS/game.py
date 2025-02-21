import math
import copy
import json
import random
import time

from pprint import pprint

#Image Visualization files
from pygame.locals import *
import pygame.surfarray as surfarray
import pygame


import numpy as np
import scipy.io as sio

from PIL import Image, ImageDraw


class Board(object):
    """
    This class represents an instantaneous board
    state of a game in progress. Should be able
    to be copied to keep track of board histories.
    """

    def __init__(self):
        raise NotImplemented

    def get_legal_actions(self):
        """
        Return a set of legal actions that can be
        on this current board.
        """

        raise NotImplemented

    def is_legal_action(self, action):
        """
        Returns True if the action is allowed
        to be performed on this board. False otherwise.

        params:
        action - the action to check
        """

        legal_actions = self.get_legal_actions()
        legal_actions = set([hash(act) for act in legal_actions])
        if hash(action) in legal_actions:
            return True
        return False

    def is_terminal(self):
        """
        Returns True if the state of the board
        is that of a finished game, False otherwise.
        """

        raise NotImplemented

    def reward_vector(self):
        """
        Returns the reward values for this state for each player
        as a tuple, with first player reward in 0th position,
        second player reward in 1st position, and so on.
        """

        raise NotImplemented

    def current_player_id(self):
        """
        Returns an integer id representing which player is to play next.
        """

        raise NotImplemented

    def visualize(self):
        """
        Visualize the board in some way or another.
        """

        raise NotImplemented

    def json_visualize(self):
        raise NotImplemented

    def __copy__(self):
        raise NotImplemented


class ConnectFourBoard(Board):
    """
    This class represents a Connect Four board.
    Locations are accessed via a 0-indexed coordinate system.
    Coordinate (x,y) means row y of column x.
    Bottom row is row 0.
    Being a two-player game, we have Red and Black
    moves, represented as R and B respectively.
    The symbol '-' means no piece is at that coordinate.
    """
    RED = 'R'
    BLACK = 'B'
    EMPTY = '-'
    NUM_COLS = 7
    NUM_ROWS = 6
    NUM_TO_CONNECT = 4
    RED_ID = 0
    BLACK_ID=1

    #Image Visualization Parameters:
    SPACESIZE = 20 #size of the tokens and board spaces in pixels; make it 128 by 128
    WINDOWWIDTH  = 144 #SPACESIZE * (NUM_COLS +1)# in pixels
    WINDOWHEIGHT = 144 #SPACESIZE * (NUM_ROWS +1) # pixels
    SPACESIZEX = SPACESIZE
    SPACESIZEY = SPACESIZE
    XMARGIN = 0 # int((WINDOWWIDTH - NUM_ROWS * SPACESIZE) / 4)
    YMARGIN = 0 #int((WINDOWHEIGHT - NUM_COLS * SPACESIZE) / 4)

    #Colors
    WHITE = (255, 255, 255)
    BGCOLOR = WHITE

    #Token Images
    redPath = './MCTS/4row_red.png'
    blackPath = './MCTS/4row_black.png'
    boardPath = './MCTS/4row_board.png'
    [RED_IMG,BLACK_IMG,BOARD_IMG] = [Image.open(p).resize((SPACESIZE,SPACESIZE)) for p in [redPath,blackPath,boardPath]]


    def __init__(self, state=None, turn=None):
        """
        params:
        state - the board state represented as nested lists
        turn - which color is to play next. either RED or BLACK
        """

        if state is None:
            self.state = [[ConnectFourBoard.EMPTY for j in xrange(ConnectFourBoard.NUM_ROWS)] for i in xrange(ConnectFourBoard.NUM_COLS)]
            self.turn = ConnectFourBoard.RED
            if np.random.rand() > 0.5:
              self.turn = ConnectFourBoard.RED
            else:
              self.turn = ConnectFourBoard.BLACK
        else:
            self.state = state
            self.turn = turn

        self.last_move = None
        
    def get_legal_actions(self):
        actions = []
        
        for col in xrange(len(self.state)):
            column = self.state[col]
            for row in xrange(len(column)):
                if column[row] == ConnectFourBoard.EMPTY:
                    actions.append(ConnectFourAction(self.turn, col, row))
                    break

        return actions

    def is_terminal(self):
        # if no move has been played, return False
        if self.last_move is None:
            return False

        # if someone has already won, return True
        if self._terminal_by_win():
            return True

        # if every slot is filled, then we've reached a terminal state
        for inv_row in xrange(ConnectFourBoard.NUM_ROWS):
            for col in xrange(ConnectFourBoard.NUM_COLS):
                row = ConnectFourBoard.NUM_ROWS - (inv_row + 1)
                if self.state[col][row] == ConnectFourBoard.EMPTY:
                    return False

        return True

    def _position_in_board(self, row, col):
        ''' given two integers, checks whether position is in board '''
        row_in_range = row >= 0 and row < ConnectFourBoard.NUM_ROWS
        col_in_range = col >= 0 and col < ConnectFourBoard.NUM_COLS
        return row_in_range and col_in_range

    def _terminal_by_win(self):    
        X = ConnectFourBoard.NUM_TO_CONNECT        
        col, row = self.last_move
        color = self.state[col][row]

        # vertical
        seq = self.state[col][row - (X - 1) : row + 1]
        if self._check_seq(color, seq):
            return True
        
        # horizontal
        min_col = max(0, col - (X -1))
        max_col = min(ConnectFourBoard.NUM_COLS-1, col+(X-1))
        seq = [self.state[i][row] for i in xrange(min_col, max_col+1)]
        if self._check_seq(color, seq):
            return True

        # up diagonal
        leftSpacesToCheck = -1 * min(row, col, X - 1)
        rightSpacesToCheck = min(ConnectFourBoard.NUM_COLS - col, ConnectFourBoard.NUM_ROWS - row, X)
        seq = [self.state[col+i][row+i] for i in xrange(leftSpacesToCheck, rightSpacesToCheck)]

        if self._check_seq(color, seq):
            return True

        # down diagonal
        leftSpacesToCheck = -1 * min(ConnectFourBoard.NUM_ROWS - row -1, col, X - 1)
        rightSpacesToCheck = min(ConnectFourBoard.NUM_COLS - col, row + 1, X)
        seq = [self.state[col+i][row-i] for i in xrange(leftSpacesToCheck, rightSpacesToCheck)]

        if self._check_seq(color, seq):
            return True
        
        return False

    def _check_seq(self, color, seq):
        counter = 0 
        for token in seq:
            if token == color:
                counter += 1
                if counter == ConnectFourBoard.NUM_TO_CONNECT:
                    return True
            else:
                counter = 0
        return False

    def reward_vector(self):
        if self._terminal_by_win():
            col, row = self.last_move
            color = self.state[col][row]
        
            if color == ConnectFourBoard.RED:
                return (1,-1)
            else:
                return (-1,1)

        return (0,0)

    def current_player_id(self):
        if self.turn == ConnectFourBoard.RED:
            return 0
        else:
            return 1

    def visualize(self):
        print
        for row in xrange(ConnectFourBoard.NUM_ROWS):
            row = ConnectFourBoard.NUM_ROWS - row - 1
            line = [self.state[col][row] for col in xrange(ConnectFourBoard.NUM_COLS)]
            line = ' '.join(line)
            print '{} '.format(row) + line
        print '  ' + ' '.join([str(col) for col in xrange(ConnectFourBoard.NUM_COLS)])
        print(self.visualize_image('test', True)[0][0])

    def visualize_image(self, makeCurrPlayerRed = False, imgName='NULL', saveImgFile=False):
        '''
        uses pillow to visualize image and returns a 3d np array.
        '''
        img = Image.new('RGB', (ConnectFourBoard.WINDOWWIDTH, ConnectFourBoard.WINDOWHEIGHT), color=ConnectFourBoard.BGCOLOR)
        getSpaceRectCoords = lambda x, y: (ConnectFourBoard.XMARGIN + (x * ConnectFourBoard.SPACESIZEX), 
            (y * ConnectFourBoard.SPACESIZEY) - ConnectFourBoard.YMARGIN )
        
	ri = ConnectFourBoard.RED_IMG
	bi = ConnectFourBoard.BLACK_IMG

	if makeCurrPlayerRed and self.turn != ConnectFourBoard.RED: #flip images
		ri  = ConnectFourBoard.BLACK_IMG
		bi = ConnectFourBoard.RED_IMG

        for x in xrange(ConnectFourBoard.NUM_COLS):
            for y in reversed(xrange(ConnectFourBoard.NUM_ROWS)):
                top_left = getSpaceRectCoords(x, ConnectFourBoard.NUM_ROWS - y)
                if self.state[x][y] == ConnectFourBoard.RED:
                    img.paste(ri, top_left)
                elif self.state[x][y] == ConnectFourBoard.BLACK:
                    img.paste(bi, top_left)
                img.paste(ConnectFourBoard.BOARD_IMG, top_left, ConnectFourBoard.BOARD_IMG)
        
        board_img = np.transpose(np.asarray(img, dtype=np.uint8), (1,0,2))
        if saveImgFile:
            completeImgName = imgName + ".jpeg"
            img.save(completeImgName)
            sio.savemat(completeImgName,{'board_img':board_img})

        return board_img

    def json_visualize(self):
        return {
            "board": self.state,
            "finished": self.is_terminal(),
            "player": self.current_player_id(),
        }

    def __copy__(self):
        new_state = copy.deepcopy(self.state)
        
        return ConnectFourBoard(new_state, self.turn)

class Action(object):
    """
    This class represents a generic action that
    can be performed on a certain board.
    Needs to be hashable in order
    to distinguish from other possible
    actions for the same board.
    """

    def __init__(self):
        raise NotImplemented

    def apply(board):
        """
        Applies the action to the board and returns a new board
        that results from it.
        Throws an error if action cannot be applied.
        """

        raise NotImplemented
        
    def __hash__(self):
        raise NotImplemented

    def string_abbreviation(self):
        raise NotImplemented


class ConnectFourAction(Action):
    """
    This board represents an action in Connect Four.
    The actions specifies the color of the piece
    and the coordinate of where to place it.
    """

    def __init__(self, color, col, row):
        """
        params:
        color - a string from ['R', 'B'] that represents the color of the piece
        col - integer for the column
        row - integer for the row
        """

        self.color = color
        self.col = col
        self.row = row

    def apply(self, board):
        #if not board.is_legal_action(self):
        #    raise Exception('This action is not allowed! => {}'.format(self))
        #check disabled because it's a pain when modifying game history
        new_board = copy.copy(board)
        new_board.state[self.col][self.row] = self.color

        if self.color == ConnectFourBoard.RED:
            new_board.turn = ConnectFourBoard.BLACK
        else:
            new_board.turn = ConnectFourBoard.RED

        new_board.last_move = (self.col, self.row)

        return new_board

    def __hash__(self):
        return hash((self.color, self.col, self.row))

    def __repr__(self):
        return 'ConnectFourAction(color={},col={},row={})'.format(self.color,self.col,self.row)

    def string_abbreviation(self):
        return self.color + str(self.col)


class Player(object):
    """
    This class represents a player.
    Subclasses can be human agents
    that takes input, or programs
    that run an algorithm.
    """

    def __init__(self, name):
        self.name = name

    def choose_action(self, board):
        """
        Returns an action that the player
        wishes to perform on the board.

        params:
        board -  the current board
        """

        raise NotImplemented

    def play_action(self, action, board):
        """
        Player performs an action
        on a given board.
        Returns the new board that results from it.

        params:
        action - the action that the player wishes to perform
        board - the current board
        """

        new_board = action.apply(board)

        return new_board


class HumanPlayer(Player):
    """
    A generic human player that takes
    a source function that returns some
    representation of the human's
    action.
    """

    def __init__(self, name, source):
        Player.__init__(self, name)
        self.source = source


class ConnectFourHumanPlayer(HumanPlayer):
    """
    Human player that plays Connect Four.
    """

    def choose_action(self, board):
        action = None
        
        while action is None:
            col, row = self.source() # example of how input comes from the source

            if col >= 0 and col < ConnectFourBoard.NUM_COLS and row >= 0 and row < ConnectFourBoard.NUM_ROWS:
                action = ConnectFourAction(board.turn, col, row)
            else:
                print 'Coordinate out of range: 0 <= COLUMN NUMBER <= {}, 0 <= ROW NUMBER <= {}'.format(ConnectFourBoard.NUM_COLS-1, ConnectFourBoard.NUM_ROWS-1)

            if not board.is_legal_action(action):
                print '{} is not a legal action on this board.'.format(action)
                action = None

        return action


class ComputerPlayer(Player):
    """
    A generic computer player that takes an algorithm
    as a strategy. If the algorithm is time-limited,
    as time limit can be supplied.
    """

    def __init__(self, name, algo, time_limit=None):
        Player.__init__(self, name)
        self.algo = algo
        self.time_limit = time_limit

    def choose_action(self, board):
        if self.time_limit:
            return self.algo(board, self.time_limit)
        else:
            return self.algo(board)

class PolicyPlayer(Player):
  """
  Policy-based player that takes a
  SupervisedPolicyAgent obejct and provides an action by
  predicting from the board image
  """
  def __init__(self,name,agent):
    Player.__init__(self,name)
    self.agent = agent
  
  def choose_action(self,board):
    board_img = board.visualize_image(makeCurrPlayerRed=True)
    legal_actions = board.get_legal_actions()
    
    if len(legal_actions) > 0:
        column_prob_dist = self.agent.predict_action(board_img)
        legal_column_prob_dist = [column_prob_dist[a.col] for a in legal_actions]
        action_idx = np.argmax(legal_column_prob_dist)
        return legal_actions[action_idx]
    raise IllegalArgumentException("This should never have occurred, the game is already over")

class ValuePlayer(Player):
  """
  Heuristic player that takes a SupervisedValueNetworkAgent
  object, and provides an action by picking the
  best action according to the future actions' estimated values
  """
  def __init__(self,name,agent):
    Player.__init__(self,name)
    self.agent = agent
  
  def choose_action(self, board):
    board_img = board.visualize_image(makeCurrPlayerRed=True)
    legal_actions = board.get_legal_actions()

    action_values = []
    for action in legal_actions:
        next_board_img = action.apply(board).visualize_image(makeCurrPlayerRed=True)
        action_value = self.agent.predict_value(next_board_img)
        action_values.append(action_value)
    best_idx = np.argmax(action_values)
    #for s in board.state:
    #  print s
    #print best_idx,legal_actions[best_idx].col
    return legal_actions[best_idx]
  
class Node(object):
    """
    A class that represents nodes in the MCTS tree.
    """

    def __init__(self, board, action=None, parent=None, heuristic=None,v_network_weight=0.0):
        """
        Create new node.

        params:
        board - Board object that this node represents.
        action - Incoming action that created this node. None for root.
        parent - Parent node. None for root.
        """

        self.board = board
        self.action = action
        self.parent = parent
        self.children = [] # children nodes
        self.num_visits = 0 # number of times node has been visited
        self.v_network_weight = v_network_weight
        if heuristic == None:
            self.v = 0.0
            self.q = 0.0
        else:
            self.v = heuristic(board)
            self.q = 0.0
        self.heuristic = heuristic
        
    def get_action(self):
        """
        Return associated incoming action.
        """
        
        return self.action

    def get_board(self):
        """
        Returns associated board state.
        """

        return self.board
        
    def get_parent(self):
        """
        Return parent node.
        """

        return self.parent

    def get_heuristic(self):
        """
        Return node's initial heuristic.
        """

        return self.heuristic

    def get_children(self):
        """
        Return children nodes.
        """
        
        return self.children

    def add_child(self, child):
        """
        Adds a new child node to the nodes visited children.
        """

        self.children.append(child)

    def get_num_visits(self):
        """
        Return number of time this node has been visited.
        """

        return self.num_visits

    def get_player_id(self):
        return self.board.current_player_id()

    def q_value(self):
        """
        Return the simulation Q value reward.
        """

        return self.q

    def visit(self):
        """
        Increment visit counter.
        """

        self.num_visits += 1

    def is_fully_expanded(self):
        """
        Returns True if the node has
        all possible children in self.children,
        False otherwise.
        """
        
        possible_actions = set([hash(action) for action in self.board.get_legal_actions()])
        taken_actions = set([hash(child.action) for child in self.children])
        untaken_actions = possible_actions ^ taken_actions
        
        return len(untaken_actions) == 0

    def value(self, c):
        """
        Return the UCT heuristic value of the node.

        params:
        c - exploration value. Larger values encourage exploration of tree.
        """

        exploitation_value = self.q / self.num_visits
        exploration_value = c * math.sqrt(2 * math.log(self.parent.num_visits) / self.num_visits)
        
        rho = self.v_network_weight ** self.num_visits
        #print rho,self.v
        return rho * self.v + (1-rho)*(exploitation_value + exploration_value)


class Simulation(object):
    """
    This class represents and simulates a game
    between two players.
    """

    def __init__(self, board, *players):
        self.init_board = board
        self.board = board
        self.players = players # players[0] is always red player?
        self.history = []

    def run(self, visualize=False, json_visualize=False, state_action_history=False, turn_winning_to_red = False):
        self.game_id = str(random.randint(0,3133337))

        tmp_history = [] #player id conscious        
        while not self.board.is_terminal():#last:
            old_board = self.board
            if visualize:
                self.board.visualize()
            if json_visualize:
                self.write_visualization_json()

            player_id = self.board.current_player_id()
            player = self.players[player_id] #players[0] is always red player?
        #    stime = time.time()
            action = player.choose_action(self.board)
        #    print 'Player '+ str(player_id) + ' took ' + str(time.time() - stime)
            self.board = player.play_action(action, self.board)
              
            tmp_history.append((player_id, action))
        winner = player_id # winner is the player that played last. NOTE: What if game ties?

        boardClass = self.board.__class__
        replay_board = boardClass()
        #TODO: For value network, always predict the value of red player, but he does not always win.
        if (ConnectFourBoard.RED_ID != winner) and turn_winning_to_red: # winner must be RED
            print 'switching winner to red'
            switchColor  = lambda x: ConnectFourBoard.RED if x == ConnectFourBoard.BLACK else ConnectFourBoard.BLACK
            switchPlayerID = lambda x: 1 if x == 0 else 0
            tmp_history = [(switchPlayerID(p), ConnectFourAction(switchColor(a.color), a.col, a.row)) for (p, a) in tmp_history]
        
        for (player_id, action) in tmp_history:
            old_board = replay_board
            replay_board = action.apply(old_board)
            entry = {}
            entry['reward'] = replay_board.reward_vector()
            entry['player_id'] = player_id
            entry['s_img'] = old_board.visualize_image()
            entry['action'] = action.col
            entry['sprime_img'] = replay_board.visualize_image()
            entry['terminal_board'] = 0
            entry['s'] = [[x for x in y] for y in old_board.state]
            entry['sprime'] = [[x for x in y] for y in replay_board.state]
            self.history.append(entry)
        self.history[-1]['terminal_board'] = 1

        if json_visualize:
            self.write_visualization_json()

        return self.history


    def write_visualization_json(self):
        data = self.board.json_visualize()
        data["gameId"] = self.game_id
        json_str = json.dumps(data)
        out_file = open("vis/game-state.json", "w")
        out_file.write(json_str)
        out_file.close()
