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
  sio.savemat(file_path + str(episode_num)+'.mat',\
                {'winner_train_data':winner_train_data,\
                 'loser_train_data':loser_train_data,\
                 'uct_time_limit':time_limit})
  return

def main():
  if len(sys.argv) < 2:
    raise Exception("Syntax: python %s episode_numbers" % (sys.argv[0]))
  else:
    num_of_episodes = int(sys.argv[1])

  generate_supervised_training_data(num_of_episodes)


if __name__ == '__main__':
    main()