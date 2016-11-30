from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K


#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#K.set_session(sess)

def load_supervised_training_data( train_dir ):
  train_files = os.listdir(train_dir)
  
  s_data = []
  a1 = []
  a2 = []
  sprime_data = []
  s_board_data = []
  sprime_board_data = []
  reward = []
  player_id = []
  value = []
  
  for f in train_files:
    try:
      data = sio.loadmat( train_dir+'/'+f )['winner_train_data']
      for i in range(data.shape[0]):
        if i==0:
          # skip the very first board position; too much variations
          continue 
        player_id += [data[i][0][0][0]]
        s_data += [data[i][1]]
        a1 += [data[i][2][0][0][1][0][0]] # col
        a2 += [data[i][2][0][0][2][0][0]] # row
        sprime_data += [data[i][3]]
        reward += [data[i][4]]
        s_config = data[i][5]
        sprime_config = data[i][6]

        s_board = [[-1 for j in xrange(np.shape(s_config)[0])] for i in xrange(np.shape(s_config)[1])]
        sprime_board = [[-1 for j in xrange(np.shape(s_config)[0])] for i in xrange(np.shape(s_config)[1])]
        n_col = np.shape(s_config)[1]
        n_row = np.shape(s_config)[0]
        for i in range(n_col):
          for j in range(n_row):
            if s_config[i,j] == 'R':
              s_board[i][j] = 0
            elif s_config[i,j] == 'B':
              s_board[i][j] = 1
            elif s_config[i,j] == '-':
              s_board[i][j] = -1

            if sprime_config[i,j] == 'R':
              sprime_board[i][j] = 0
            elif sprime_config[i,j] == 'B':
              sprime_board[i][j] = 1
            elif sprime_config[i,j] == '-':
              sprime_board[i][j] = -1
        s_board_data +=[s_board]
        sprime_board_data += [sprime_board]

        if player_id[-1] == 0:
          # value = end reward
          value += [data[-1][4][0][0]]
        else:
          value += [data[-1][4][0][1]]
      if len(a1) > 22000:
        return np.asarray(s_data),np.asarray(a1),np.asarray(a2),\
        np.asarray(player_id),np.asarray(reward),np.asarray(value),s_board_data,sprime_board_data
    except:
      # some files corrupted
      continue
  return np.asarray(s_data),np.asarray(a1),np.asarray(a2),\
        np.asarray(player_id),np.asarray(reward),np.asarray(value),s_board_data,sprime_board_data

def main():
  train_dataset_dir = '../dataset/'
  s_data,a1,a2,player_id,_,_,_,_ = load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedPolicyAgent((144,144,3),8)
  rl_agent.update_supervised_policy(s_data,a1,player_id)
  rl_player = game.RLPlayer('algo_1', rl_agent)
  test_policy_vs_MCTS(rl_player)

if __name__ == '__main__':
    main()
