from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS, test_policy_scenarios
import tensorflow as tf
from keras import backend as K

import MCTS.mcts as mcts
def main():
  rl_agent = SupervisedPolicyAgent((144,144,3),7)
  rl_agent.load_train_results()  
  rl_player = game.PolicyPlayer('algo_1', rl_agent)
  print("Player Ready")
  #score,episode = test_policy_vs_MCTS(rl_player,verbose=True)
  test_policy_scenarios(rl_player, True)


if __name__ == '__main__':
    main()
