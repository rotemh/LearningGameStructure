from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K
from learn_sup_policy_network import load_supervised_training_data


def main():
  rl_agent = SupervisedPolicyAgent((144,144,3),8)
  rl_agent.load_train_results()  
  rl_player = game.RLPlayer('algo_1', rl_agent)

  s_data,a1,_,player_id,_,_,_,_ = load_supervised_training_data('../dataset')
  predictions = rl_agent.predict_action(s_data,player_id)
  print predictions
  print np.sum( np.argmax(predictions,1) == a1 ) / np.shape(s_data)[0]
  import pdb;pdb.set_trace()
  
  
  test_policy_vs_MCTS(rl_player,mcts_times=[0.5],verbose=True)

if __name__ == '__main__':
    main()
