from MCTS.sim import *
from RL.SupervisedPolicy import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K
from learn_sup_policy_network import load_supervised_training_data


def amcts(policy_agent, value_agent, time_limit):

  value_player = RLPlayer('value_player', value_agent)
  policy_player = RLPlayer('policy_player', policy_agent)

  def uct_heuristic(board):
    return 0
    board_image = board.visualize_image()
    action_values = value_player.predict_value()
    return np.max(action_values)

  def default_heuristic(board):
    return policy_player.choose_action(board)

  algo = uct_with_heuristics(time_limit, uct_heuristic, default_heuristic)
  return algo

def main():
  policy_agent = SupervisedPolicyAgent((144,144,3),8)
  policy_agent.load_train_results()
  value_agent = SupervisedValueNetworkAgent((144,144,3),8)
  #value_agent.load_train_results()
  amcts_algo = amcts(policy_agent,value_agent,1)
  acmts_player = game.RLPlayer('amcts', amcts_algo)
  
  test_policy_vs_MCTS(amcts,mcts_times=[0.5],verbose=True)

if __name__ == '__main__':
    main()
