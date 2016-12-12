from MCTS.sim import *
from RL.SupervisedPolicy import *
from RL.SupervisedValueNetwork import *
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
  test_policy_scenarios(rl_player, True)
  score,episode = test_policy_vs_MCTS(rl_player,verbose=True)
  """
  rl_agent = SupervisedValueNetworkAgent((144,144,3))
  rl_agent.load_train_results()  
  rl_player =game.ValuePlayer('value_only', rl_agent)
  score,episode = test_policy_vs_MCTS(rl_player,verbose=True)
  """

if __name__ == '__main__':
    main()
