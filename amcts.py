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

  def __init__(self, name, time_limit, policy_agent = None, value_agent = None, rollout_randomness=0.3, value_prescaling=5.):
    self.policy_agent = policy_agent
    self.value_agent = value_agent
    self.rollout_randomness = rollout_randomness # % of the time rollout step is random, vs. policy based
    self.value_prescaling = value_prescaling # Multiplies the strength of the value-agent prediction
    ComputerPlayer.__init__(self, name, self.amcts_algo(), time_limit)

  def amcts_algo(self):
    def uct_heuristic(board):
      if self.value_agent == None:
        return 0

      board_image = board.visualize_image()
      return self.value_prescaling*self.value_agent.predict_value(board_image)
  
    def default_heuristic(board):
      actions = board.get_legal_actions()

      if self.policy_agent == None or np.random.rand < np.rollout_randomness:      
        action = np.random.choice(list(actions))
        return action

      board_image = board.visualize_image()
      column_prob_dist = self.policy_agent.predict_action(board_image)
      legal_column_prob_dist = [column_prob_dist[action.col] for action in actions]
      action_idx = np.argmax(legal_column_prob_dist)
      return actions[action_idx]

    algo = lambda board, time_limit: uct_with_heuristics(board, time_limit, uct_heuristic, default_heuristic)
    return algo


def main():
  policy_agent = SupervisedPolicyAgent((144,144,3),7)
  policy_agent.load_train_results()
  value_agent = SupervisedValueNetworkAgent((144,144,3))
  value_agent.load_train_results()
  #amcts_algo = amcts(policy_agent,value_agent,0.5)
  #amcts_player = game.ComputerPlayer('amcts', amcts_algo)
  amcts_player = AMCTSPlayer('amcts', 2, policy_agent, value_agent)

  test_policy_vs_MCTS(amcts_player,verbose=True)

if __name__ == '__main__':
    main()
