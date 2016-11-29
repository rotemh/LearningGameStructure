from MCTS.sim import *
from RL.SupervisedValueNetwork import *
from learn_sup_policy_network import load_supervised_training_data
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS

def main():
  train_dataset_dir = '../dataset/'
  s_data,_,_,player_id,_,value= load_supervised_training_data(train_dataset_dir)
  import pdb;pdb.set_trace()
  
  rl_agent = SupervisedValueNetworkAgent((144,144,3),8)
  rl_agent.update_v_network(s_data,player_id,value)

if __name__ == '__main__':
    main()
