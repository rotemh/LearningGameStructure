# Generates Connect Four images
import globalVars as gv 
from tokens import myTokens as tokens
from players import Human, Computer

import random, sys, pygame, copy 
from pygame.locals import *

def main():
	#Set up display environment
	global window, Empty

	Empty = None #default value for board rep with no piece

	pygame.init()
	window = pygame.display.set_mode((gv.WindowWidth, gv.WindowHeight))
	pygame.display.set_caption('Connect X')

	#Create Players
	player1 = Human(tokens.Red, "p1")
	player2 = Computer(tokens.Black, "p2")

	#Run game:
	winner = runGame(player1, player2)
	print(winner.name)

def runGame(player1, player2, gameNum = 0, saveImages=True):
	'''
	Runs a game between two players. Saves each image of the 
	game if saveImages is set to true.
	'''

	#Check Valid Players
	checkValidPlayer = lambda p: isinstance(p, Human) or isinstance(p, Computer)
	assert checkValidPlayer(player1) and checkValidPlayer(player2)
	
	#Turns
	turn = player1 if random.randint(0, 1) == 0 else player2
	switchTurn = lambda: player1 if turn == player2 else player2

	#Board
	board = getNewBoard()
	
	getPlayerFromToken = lambda w: player1 if w == player1.token.name else player2
	
	winner = None
	for i in xrange(gv.BoardHeight * gv.BoardWidth):
		#Make a move
		column = turn.getMove(board)
		action = makeMove(board, turn.token.name, column)

		#save image of board
		if saveImages:
			createBoardImage(board)
			pygame.display.update()
			imgName = gv.saveFolder  + "Game" + str(gameNum) + "_Step" + str(i) + ".jpeg"
			pygame.image.save(window, imgName)

		#Did the current player now win?
		if isWinner(board, turn.token.name):
			winner = turn
			break

		#Tie?
		if isBoardFull(board):
			break

		#switch players
		turn = switchTurn()

	return winner

'''
def getHumanMove(board):
	playerInp = raw_input("Enter column here: ")
	column = int(playerInp)
	while not checkValidMove(board, column):
		playerInp = raw_input("Invalid column. Try entering a column again: ")
		column = int(playerInp)
	return column


def getComputerMove(board, AI=None):
	if AI is None:
		return getDummyMove(board)
	else:
		return AI(board)

def getDummyMove(board):
	column = random.randint(0, 6)
	while not checkValidMove(board, column):
		column = random.randint(0, 6)
	return column
'''

def isWinner(board, token):
	# horizontal
    for x in xrange(gv.BoardWidth - (gv.ConnectX - 1)):
        for y in xrange(gv.BoardHeight):
        	horizWin = True
        	for i in xrange(gv.ConnectX):
        		if board[x + i][y] != token:
        			horizWin = False
        			break
        	if horizWin:
        		return True

    # vertical
    for x in xrange(gv.BoardWidth):
        for y in xrange(gv.BoardHeight - (gv.ConnectX - 1)):
        	vertWin = True
        	for i in xrange(gv.ConnectX):
        		if board[x][y + i] != token:
        			vertWin = False
        			break
        	if vertWin:
        		return True

    # / diagonals
    for x in xrange(gv.BoardWidth - (gv.ConnectX - 1)):
        for y in xrange((gv.ConnectX - 1), gv.BoardHeight):
        	diagWin = True
        	for i in xrange(gv.ConnectX):
        		if board[x + i][y - i] != token:
        			diagWin = False
        			break
        	if diagWin:
        		return True
    # \ diagonals 
    for x in range(gv.BoardWidth - (gv.ConnectX - 1)):
        for y in range(gv.BoardHeight - (gv.ConnectX - 1)):
        	diagWin = True
        	for i in xrange(gv.ConnectX):
        		if board[x + i][y + i] != token:
        			diagWin = False
        			break
        	if diagWin:
        		return True
    return False

def isBoardFull(board):
    '''
    Returns True if there are no empty spaces anywhere 
    on the board.
    '''
    for x in xrange(gv.BoardWidth):
        for y in xrange(gv.BoardHeight):
            if board[x][y] == Empty:
                return False
    return True
'''
def checkValidMove(board, column):
	if column < 0 or column >= BoardWidth or board[column][0] != Empty:
		return False
	return True
'''

def makeMove(board, token, column):
	'''
	Makes move if possible and returns action
	as dictionary of form:
	{'x': column, 'y': row, 'color':token})
	'''
	lowest = getLowestEmptySpace(board, column)
	if lowest is not None:
		board[column][lowest] = token
		return {'x': column, 'y': lowest, 'color':token}
	else:
		return {'x': None, 'y': None, 'color': Empty}

def getLowestEmptySpace(board, column):
	'''
	Given a board and column, return the lowest empty space
	'''
	height = len(board[column])
	for index, value in enumerate(reversed(board[column])):
	    if value == Empty:
	        return height - index - 1
	return None

def getNewBoard():
	'''
	implemented as an array of column arrays, with
	Empty in the empty space, and 0,0 referring to
	the top left corner.
	'''
	board = []
	for x in xrange(gv.BoardWidth):
		board.append([Empty] * gv.BoardHeight)
	return board

#Image generation and manipulation
def createBoardImage(board, latestAction=None):
	window.fill(gv.BgColor)

	spaceRect = pygame.Rect(0, 0, gv.SpaceSize, gv.SpaceSize)
	getSpaceRectCoords = lambda x, y: (gv.Xmargin + (x * gv.SpaceSize), gv.Ymargin + (y * gv.SpaceSize))
	for x in xrange(gv.BoardWidth):
		for y in xrange(gv.BoardHeight):
			spaceRect.topleft = getSpaceRectCoords(x, y)
			if tokens.isPlayerToken(board[x][y]):
				img = tokens.getImage(board[x][y])
				window.blit(img, spaceRect)

	# draw the extra token
	if latestAction != None:
		img = tokens.getImage(latestAction['color'])
		window.blit(img, (latestAction['x'], latestAction['y'], gv.SpaceSize, gv.SpaceSize))

	# draw board over the tokens
	for x in range(gv.BoardWidth):
		for y in range(gv.BoardHeight):
			spaceRect.topleft = getSpaceRectCoords(x, y)
			window.blit(tokens.Board.img, spaceRect)



main()