from MCTS.sim import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS

import tensorflow as tf

def load_supervised_training_data( train_dir ):
  train_files = os.listdir(train_dir)
  
  s_data = []
  a = []
  sprime_data = []
  s_board_data = []
  sprime_board_data = []
  reward = []
  player_id = []
  value = []
  
  for f in train_files:
    if f[f.find('.'):] != '.p':
      continue
    try:
      data = pickle.load(open( train_dir+'/'+f))
      black = data['black_player_data']
      data = data['red_player_data']
      for i in range(len(data)):
        if i==0:
          # skip the very first board position; too much variations
          continue 
        s_data.append(data[i]['s_img'])
        if data[-1]['terminal_board']:
          # value = end reward
          value.append(data[-1]['reward'][0])
        else:
          assert(black[-1]['terminal_board'])
          value.append(black[-1]['reward'][0])
      if len(a) > 52000:
        return np.asarray(s_data),np.asarray(value)
    except ValueError:
      print f + ' is corrupted'
  return np.asarray(s_data),np.asarray(value)

def main():
  train_dataset_dir = '/home/beomjoon/LearningGameStructure/v_dataset/'
  s_data,value = load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedValueNetworkAgent((144,144,3))
  rl_agent.update_v_network(s_data,value)

if __name__ == '__main__':
    main()
