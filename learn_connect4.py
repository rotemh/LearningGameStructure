from MCTS.sim import *
from RL.SupervisedPolicy import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS

def load_supervised_training_data( train_dir, singleActionToDoubleAction = True ):
  '''
  loading double action moves. If you want to load from the single move dataset, set the 
  indicator boolean to true. 
  '''
  train_files = os.listdir(train_dir)
  
  s_data = []
  a = []
  sprime_data = []
  for f in train_files:
    try:
      if singleActionToDoubleAction:
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
      else:  
        data = sio.loadmat( train_dir+'/'+f )['winner_train_data']
        for i in range(data.shape[0]):
          s_data += [data[i][1]]
          a += [data[i][2][0][0][1][0][0]] # col
          sprime_data += [data[i][3]]
          #reward += [data[i][4]]
          #player_id += [data[i][0]]
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
  train_dataset_dir = './dataset/'
  s_data,a1,a2,player_id = load_supervised_training_data(train_dataset_dir)
  rl_agent = SupervisedPolicyAgent((144,144,3),8)
  rl_agent.update_supervised_policy(s_data,a1,player_id)
  rl_player = game.RLPlayer('algo_1', rl_agent)
  test_policy_vs_MCTS(rl_player)

if __name__ == '__main__':
    s,a,sp,r = load_supervised_training_data('./dataset')
