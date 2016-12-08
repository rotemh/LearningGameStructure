from MCTS.sim import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from learn_sup_policy_network import load_supervised_training_data
from tester import test_policy_vs_MCTS

def main():
  train_dataset_dir = '/home/beomjoon/LearningGameStructure/dataset/'
  #import pdb;pdb.set_trace()
  s_data,a,player_id,r,value,_,_= load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedValueNetworkAgent((144,144,3),7)
  rl_agent.update_v_network(s_data,player_id,value)

if __name__ == '__main__':
    main()
