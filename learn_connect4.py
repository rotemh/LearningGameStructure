from MCTS.sim import *
from RL.RLAgent import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os
from tester import test_policy_vs_MCTS

def generate_supervised_training_data(num_episodes, time_limit=0.5, file_path='/media/beomjoon/My Passport/vision_project/supervised_data/train_data'):
  train_data= []
  for x in xrange(num_episodes): 
    print x
    episode = generate_uct_game(time_limit)
    win_player_id = np.argmax(episode[-1][-1])
    winner_train_data = [e for e in episode if e[0] == win_player_id]
    loser_train_data = [e for e in episode if e[0] != win_player_id]
    sio.savemat(file_path + str(x)+'.mat',{'winner_train_data':winner_train_data,'loser_train_data':loser_train_data,\
                  'uct_time_limit':time_limit})

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
      data = sio.loadmat( train_dir+'/'+f )['train_data']
      for i in range(data.shape[0]):
        s_data += [data[i][1]]
        a1 += [data[i][2][0][0][1][0][0]] # col
        a2 += [data[i][2][0][0][2][0][0]] # row
        sprime_data += [data[i][3]]
        reward += [data[i][4]]
        player_id += [data[i][0]]
    except:
      # some files corrupted
      continue
  return np.asarray(s_data),np.asarray(a1),np.asarray(a2),np.asarray(player_id)

def main():
  #generate_supervised_training_data(100000)
  s_data,a1,a2,player_id = load_supervised_training_data('./train_data')
  rl_agent = ReinforcementLearningAgent((144,144,3),8)
  rl_agent.update_supervised_policy(s_data,a1,player_id)
#  episode = generate_custom_policy_game(rl_agent,rl_agent)
  rl_player = game.RLPlayer('algo_1', rl_agent)
  import pdb;pdb.set_trace()
  test_policy_vs_MCTS(rl_player)

if __name__ == '__main__':
    main()
