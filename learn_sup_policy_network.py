from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS
import tensorflow as tf
from keras import backend as K
from load_data import yield_sup_policy_data
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
        player_id.append(data[i]['player_id'])
        s_data.append(data[i]['s_img'])
        a.append(data[i]['action']) # col stored only
        sprime_data.append(data[i]['sprime_img'])
        reward.append(data[i]['reward'])
        s_config = data[i]['s']
        sprime_config = data[i]['sprime']

	token = lambda(x) : 0 if x=='R' else (1 if x =='B' else -1)	
        s_board = [ [token(j) for j in col] for col in s_config ]
        sprime_board = [[token(j) for j in col] for col in sprime_config]
        s_board_data.append(s_board)
        sprime_board_data.append(sprime_board)

        if player_id[-1] == 0:
          # value = end reward
          value.append(data[-1]['reward'][0])
        else:
          value.append(data[-1]['reward'][1])
      if len(a) > 50000:
        print len(a)
        return np.asarray(s_data),np.asarray(a),\
          np.asarray(player_id),np.asarray(reward),np.asarray(value),\
          np.asarray(s_board_data),np.asarray(sprime_board_data)
    except ValueError:
      print f + ' is corrupted'
 
  return np.asarray(s_data),np.asarray(a),\
    np.asarray(player_id),np.asarray(reward),np.asarray(value),\
    np.asarray(s_board_data),np.asarray(sprime_board_data)

def main():
#  train_dataset_dir = '/home/beomjoon/LearningGameStructure/dataset/'
#  s_data,a,_,_,_,_,_ = load_supervised_training_data(train_dataset_dir)
  s_data = pickle.load(open('./p_dataset_s.p'))
  a = pickle.load(open('./p_dataset_a.p'))
  print 'loading complete'
  rl_agent = SupervisedPolicyAgent((144,144,3),7)
  rl_agent.update_supervised_policy(s_data,a)

if __name__ == '__main__':
    main()
