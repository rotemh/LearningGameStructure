from MCTS.sim import *
from RL.RLAgent import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os


def main():
	agent = ReinforcementLearningAgent((144,144,3),7)
	agent.train_supervised_Q()

if __name__ == '__main__':
    main()