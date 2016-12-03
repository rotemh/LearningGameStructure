from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K
import pickle

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#K.set_session(sess)

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
      data = pickle.load(open( train_dir+'/'+f))['winner_train_data']
      for i in range(len(data)):
        if i==0:
          # skip the very first board position; too much variations
          continue 
        player_id += [data[i]['player_id']]
        s_data += [data[i]['s_img']]
        a += [data[i]['action'].col] # col
        sprime_data += [data[i]['sprime_img']]
        reward += [data[i]['reward']]
        s_config = [data[i]['s']]
        sprime_config = [data[i]['sprime']]

        s_board = [ [-1 for j in xrange(np.shape(s_config[-1])[0])] for i in xrange(np.shape(s_config[-1])[1]) ]
        sprime_board = [[-1 for j in xrange(np.shape(s_config[-1])[0])] for i in xrange(np.shape(s_config[-1])[1])]
        n_col = np.shape(s_config[-1])[1]
        n_row = np.shape(s_config[-1])[0]
        for i in range(n_col):
          for j in range(n_row):
            if s_config[-1][i][j] == 'R':
              s_board[i][j] = 0
            elif s_config[-1][i][j] == 'B':
              s_board[i][j] = 1
            elif s_config[-1][i][j] == '-':
              s_board[i][j] = -1

            if sprime_config[-1][i][j] == 'R':
              sprime_board[i][j] = 0
            elif sprime_config[-1][i][j] == 'B':
              sprime_board[i][j] = 1
            elif sprime_config[-1][i][j] == '-':
              sprime_board[i][j] = -1
        s_board_data +=[s_board]
        sprime_board_data += [sprime_board]

        if player_id[-1] == 0:
          # value = end reward
          value += [data[-1]['reward'][0]]
        else:
          value += [data[-1]['reward'][1]]
      if len(a) > 220:
        return np.asarray(s_data),np.asarray(a),\
          np.asarray(player_id),np.asarray(reward),np.asarray(value),\
          np.asarray(s_board_data),np.asarray(sprime_board_data)
    except ValueError:
      print f + ' is corrupted'
 
  return np.asarray(s_data),np.asarray(a),\
    np.asarray(player_id),np.asarray(reward),np.asarray(value),\
    np.asarray(s_board_data),np.asarray(sprime_board_data)

def main():
  train_dataset_dir = './dataset/'
  s_data,a,player_id,_,_,_,_ = load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedPolicyAgent((144,144,3),7)
  rl_agent.update_supervised_policy(s_data,a,player_id)
  rl_player = game.RLPlayer('algo_1', rl_agent)
  test_policy_vs_MCTS(rl_player)

if __name__ == '__main__':
    main()
