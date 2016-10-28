import globalVars
import pygame


class Token(object):
	"""Token are the coin types"""
	def __init__(self, name, imgPath):
		self.name = name
		self.imgPath = imgPath
		img = pygame.image.load(imgPath)
		self.img = pygame.transform.smoothscale(img, (globalVars.SpaceSize, globalVars.SpaceSize))

class Tokens(object):
	"""docstring for ClassName"""
	def __init__(self):
		Tokens.Red = Token('Red', '4row_red.png')
		Tokens.Black = Token('Black', '4row_black.png')
		Tokens.Board = Token('Board', '4row_board.png')
		Tokens.Empty = None
		Tokens.PlayerTokens = [Tokens.Red, Tokens.Black]

	@staticmethod
	def getImage(name):
		for token in [Tokens.Red, Tokens.Black, Tokens.Board]:
			if name == token.name:
				return token.img
		return None

	@staticmethod
	def isPlayerToken(name):
		for token in Tokens.PlayerTokens:
			if name == token.name:
				return True
		return False

		
myTokens = Tokens()