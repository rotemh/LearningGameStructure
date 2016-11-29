from MCTS.sim import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS

def load_supervised_training_data( train_dir ):
  train_files = os.listdir(train_dir)
  s_data = []
  a1 = []
  a2 = []
  sprime_data = []
  reward = []
  player_id = []
  value = []
  
  for f in train_files:
    try:
      player_name = ['winner_train_data','loser_train_data']
      data = sio.loadmat( train_dir+'/'+f )['winner_train_data']
      winner_player_id = data[0][0]
      for p in player_name:
        data = sio.loadmat( train_dir+'/'+f )[p]
        for i in range(data.shape[0]):
          if i==0:
            # skip the very first board position; too much variations
            continue 
          s_data += [data[i][1]]
          a1 += [data[i][2][0][0][1][0][0]] # col
          a2 += [data[i][2][0][0][2][0][0]] # row
          sprime_data += [data[i][3]]
          player_id += [data[i][0]]
          reward += [data[i][4]]

          if player_id[-1] == winner_player_id:
            value += [1]
          else:
            value += [-1]
        if len(a1) > 22000:
          return np.asarray(s_data),np.asarray(a1),np.asarray(a2),\
                np.asarray(player_id),np.asarray(reward),np.asarray(value)
    except:
      # some files corrupted
      continue
  return np.asarray(s_data),np.asarray(a1),np.asarray(a2),\
        np.asarray(player_id),np.asarray(reward),np.asarray(value)

def main():
  train_dataset_dir = '../dataset/'
  s_data,_,_,player_id,_,value= load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedValueNetworkAgent((144,144,3),8)
  rl_agent.update_v_network(s_data,player_id,value)

if __name__ == '__main__':
    main()
