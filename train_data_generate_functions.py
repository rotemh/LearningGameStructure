from MCTS.sim import *
#from RL.RLAgent import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
import sys
import threading
from tester import test_policy_vs_MCTS
from Queue import Queue
from threading import Thread
from RL.SupervisedPolicy import SupervisedPolicyAgent
import pickle 

def generate_policy_training_data(episode_num, time_limit=1, file_path='./dataset/'):
  episode = generate_uct_game(time_limit)
  if episode[-1]['terminal_board'] and (episode[-1]['reward'][0] is not episode[-1]['reward'][1]):
    win_player_id = np.argmax( episode[-1]['reward'] )
  else:
    return
  train_data = {}
  winner_train_data = [e for e in episode if e['player_id'] == win_player_id]
  loser_train_data = [e for e in episode if e['player_id'] != win_player_id]
  train_data['winner_train_data']=winner_train_data
  train_data['loser_train_data']=loser_train_data
  pickle.dump( train_data,open(file_path+str(episode_num)+'.p','wb'))
  sio.savemat(file_path + str(episode_num)+'.mat',\
                {'winner_train_data':winner_train_data,\
                 'loser_train_data':loser_train_data,\
                 'uct_time_limit':time_limit})
  return

def generate_v_training_data(episode_num, rl_player,time_limit=0.05, file_path='./v_dataset/'):
  episode = generate_custom_vs_uct_game(rl_player,time_limit)
  
  # get each player's data
  red_player_id = 0
  black_player_id = 1
  train_data = {}
  red_player_data = [e for e in episode if e['player_id'] == red_player_id]
  black_player_data = [e for e in episode if e['player_id'] == black_player_id]

  train_data['red_player_data'] = red_player_data
  train_data['black_player_data'] = black_player_data
  pickle.dump( train_data,open(file_path+str(episode_num)+'.p','wb'))
  sio.savemat(file_path + str(episode_num)+'.mat',\
                {'train_data':train_data,\
                 'uct_time_limit':time_limit})
  return

