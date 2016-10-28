from __future__ import print_function
import random

import globalVars
from tokens import myTokens as tokens

class Player(object):
	"""docstring for Player"""
	def __init__(self, token, name, moveFunction):
		self.token = token
		self.name = name
		self.moveFunction = moveFunction 

	def getMove(self, board):
		return self.moveFunction(self, board)

	@staticmethod
	def checkValidMove(board, column):
		if column < 0 or column >= globalVars.BoardWidth or board[column][0] != tokens.Empty:
			return False
		return True



class Human(Player):
	"""human or computer?"""
	def __init__(self, token, name='human'):
		Player.__init__(self, token, name, getHumanMove )	

def getHumanMove(self, board):
	playerInp = raw_input("Enter column here: ")
	column = int(playerInp)
	while not Player.checkValidMove(board, column):
		playerInp = raw_input("Invalid column. Try entering a column again: ")
		column = int(playerInp)
	return column




class Computer(Player):
	"""human or computer?"""
	def __init__(self, token, name='computer', moveFunction=None):
		if moveFunction is None:
			moveFunction = getDummyMove
		Player.__init__(self, token, name, moveFunction )	



def getDummyMove(self, board):
	'''
	randomly pick a valid move
	'''
	column = random.randint(0, 6)
	while not Player.checkValidMove(board, column):
		column = random.randint(0, 6)
	return column

defaultHuman = Human(tokens.Red)
defaultComputer = Computer(tokens.Black)