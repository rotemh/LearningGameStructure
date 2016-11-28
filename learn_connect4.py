from MCTS.sim import *
#from RL.RLAgent import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
import sys
import threading
from tester import test_policy_vs_MCTS

def generate_supervised_training_data(episode_num, time_limit=0.5, file_path=''):
  train_data= []
  episode = generate_uct_game(time_limit)
  win_player_id = np.argmax(episode[-1][-1])
  winner_train_data = [e for e in episode if e[0] == win_player_id]
  loser_train_data = [e for e in episode if e[0] != win_player_id]
  sio.savemat(file_path + str(episode_num)+'.mat',{'winner_train_data':winner_train_data,'loser_train_data':loser_train_data,\
                'uct_time_limit':time_limit})
  return

def load_supervised_training_data( train_dir ):
  '''
  loading double action moves from single action move
  set. 
  '''
  train_files = os.listdir(train_dir)
  
  s_data = []
  a = []
  sprime_data = []
  for f in train_files:
    try:
      data = sio.loadmat( train_dir+'/'+f )['train_data']
      num_of_moves = data.shape[0]
      winning_player = data[num_of_moves -1][0]
      reward = [0 for i in range(winning_player, num_of_moves, 2)]
      reward[-1] = 1
      
      for i in range(winning_player, num_of_moves, 2):
        s_data += [data[i][1]]
        a += [data[i][2][0][0][1][0][0]] # col
        sprime_data += [data[i+1][3]]
        #reward += [data[i][4][0][winning_player][0][0]]
        #player_id += [data[i][0][0][0]]
    except:
      # some files corrupted
      continue
  return np.asarray(s_data),np.asarray(a),np.asarray(sprime_data),np.asarray(reward)

def parse_arg_to_generate_data():
  if len(sys.argv) < 2:
    raise Exception("Syntax: python %s episode_numbers" % (sys.argv[0]))
  else:
    num_of_episodes = int(sys.argv[1])

  generate_supervised_training_data(num_of_episodes)

def main():
  pass

if __name__ == '__main__':
    s,a,sp,r = load_supervised_training_data('./dataset')
