from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K

import MCTS.mcts as mcts
def main():
  rl_agent = SupervisedPolicyAgent((144,144,3),7)
  rl_agent.load_train_results()  
  rl_player = game.RLPlayer('algo_1', rl_agent)
  score,episode = test_policy_vs_MCTS(rl_player,verbose=True)
  sio.savemat('./policyNetworkGameResults/results.mat',{'score':score,'episodes':episode})
  

if __name__ == '__main__':
    main()
