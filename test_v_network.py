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
from amcts import *

import MCTS.mcts as mcts
def main():
  """
  rl_agent = SupervisedPolicyAgent((144,144,3),7)
  rl_agent.load_train_results()  
  rl_player = game.PolicyPlayer('algo_1', rl_agent)
  print("Player Ready")
  test_policy_scenarios(rl_player, True)
  score,episode = test_policy_vs_MCTS(rl_player,verbose=True)
  """
  rl_agent = SupervisedValueNetworkAgent((144,144,3))
  rl_agent.load_train_results()  
  pol = SupervisedPolicyAgent((144,144,3),7)
  pol.load_train_results()  
#  rl_player =game.ValuePlayer('value_only', rl_agent)

  amcts_v_1 = AMCTSPlayer('AMCTS_v_p1s', 0.1, value_agent=rl_agent,v_network_weight=0.5)
  score,episode = test_policy_vs_MCTS(amcts_v_1,verbose=True)

if __name__ == '__main__':
    main()
