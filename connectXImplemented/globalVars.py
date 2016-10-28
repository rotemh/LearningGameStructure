#Global Vars:
BoardWidth = 7  # how many spaces wide 
BoardHeight = 6 # how many spaces tall 
ConnectX = 4 #must connect four to win
assert BoardWidth >= ConnectX,  'Board is not wide enough to accomodate game.'
assert BoardHeight >= ConnectX, 'Board is not tall enough to accomodate game.'

#AI Vars:
LookAheadXSteps = 2

#Visualization Vars
SpaceSize = 50 #size of the tokens and board spaces in pixels
WindowWidth  = 640 # in pixels
WindowHeight = 480 # pixels
Xmargin = int((WindowWidth - BoardWidth * SpaceSize) / 2)
Ymargin = int((WindowHeight - BoardHeight * SpaceSize) / 2)

#Colors
white = (255, 255, 255)
BgColor = white

#Save Images in Folder Called
saveFolder = "images/"
