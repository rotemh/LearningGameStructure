from MCTS.sim import *
from RL.SupervisedPolicy import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
from MCTS.mcts import *
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K
from learn_sup_policy_network import load_supervised_training_data

class AMCTSPlayer(ComputerPlayer):
  """
  A player which uses an MCTS algorithm
  enhanced by a value network and a Policy
  network to more efficiently explore the 
  search tree.
  """

  def __init__(self, name, policy_agent, value_agent, time_limit=None):
    self.policy_agent = policy_agent
    self.value_agent = value_agent
    ComputerPlayer.__init__(self, name, self.amcts_algo(), time_limit)

  def amcts_algo(self):
    def uct_heuristic(board):
      board_image = board.visualize_image()
      return self.value_agent.predict_value(board_image)
  
    def default_heuristic(board):
      actions = board.get_legal_actions()
      action = np.random.choice(list(actions))
      return action
      # Currently ignoring default heuristic
      return self.policy_player.choose_action(board)

    algo = lambda board, time_limit: uct_with_heuristics(board, time_limit, uct_heuristic, default_heuristic)
    return algo


def main():
  policy_agent = SupervisedPolicyAgent((144,144,3),7)
  policy_agent.load_train_results()
  value_agent = SupervisedValueNetworkAgent((144,144,3))
  value_agent.load_train_results()
  #amcts_algo = amcts(policy_agent,value_agent,0.5)
  #amcts_player = game.ComputerPlayer('amcts', amcts_algo)
  amcts_player = AMCTSPlayer('amcts', policy_agent, value_agent, 3)

  test_policy_vs_MCTS(amcts_player,verbose=True)

if __name__ == '__main__':
    main()
