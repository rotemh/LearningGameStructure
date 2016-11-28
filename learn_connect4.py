from MCTS.sim import *
from RL.SupervisedPolicy import *
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
  for f in train_files:
    try:
      data = sio.loadmat( train_dir+'/'+f )['winner_train_data']
      for i in range(data.shape[0]):
        s_data += [data[i][1]]
        a1 += [data[i][2][0][0][1][0][0]] # col
        a2 += [data[i][2][0][0][2][0][0]] # row
        sprime_data += [data[i][3]]
        reward += [data[i][4]]
        player_id += [data[i][0]]
        print len(a1)
        if len(a1) > 22000:
          return np.asarray(s_data),np.asarray(a1),np.asarray(a2),np.asarray(player_id)
    except:
      # some files corrupted
      continue
  return np.asarray(s_data),np.asarray(a1),np.asarray(a2),np.asarray(player_id)

def main():
  train_dataset_dir = './dataset/'
  s_data,a1,a2,player_id = load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedPolicyAgent((144,144,3),8)
  rl_agent.update_supervised_policy(s_data,a1,player_id)
  rl_player = game.RLPlayer('algo_1', rl_agent)
  test_policy_vs_MCTS(rl_player)

if __name__ == '__main__':
    main()
