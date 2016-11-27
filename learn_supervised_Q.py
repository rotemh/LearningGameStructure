from MCTS.game import *
from MCTS.sim import *
from RL.SupervisedQAgent import *
import tester 
import numpy as np
import scipy.io as sio
import os


def main():
	agent = SupervisedQAgent((144,144,3),7)
	agent.train()
	player = RLPlayer("supervisedq",agent)
	tester.test_policy_vs_MCTS(player,verbose=True)

if __name__ == '__main__':
    main()