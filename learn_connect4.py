from MCTS.sim import *
from RL.RLAgent import *
from MCTS.game import * 
import numpy as np
import scipy.io as sio
import os

def generate_supervised_training_data(num_episodes):
  train_data= []
  for x in xrange(num_episodes): 
    print x
    episode = generate_uct_game(time_limit=0.5)
    win_player_id = np.argmax(episode[-1][-1])
    winner_train_data = [e for e in episode if e[0] == win_player_id]
    loser_train_data = [e for e in episode if e[0] != win_player_id]
    sio.savemat('/media/beomjoon/My Passport/vision_project/supervised_data/train_data' \
                  + str(x)+'.mat',{'winner_train_data':winner_train_data,'loser_train_data':loser_train_data})

def load_supervised_training_data( train_dir ):
  train_files = os.listdir(train_dir)
  
  s_data = []
  a = []
  sprime_data = []
  reward = []
  for f in train_files:
    try:
      data = sio.loadmat( train_dir+'/'+f )['train_data']
      s_data += [data[0][1]]
      a += [data[0][2][0][0][1][0][0]] # col
      sprime_data += [data[0][3]]
      reward += [data[0][4]]
    except:
      # some files corrupted
      continue
  return np.asarray(s_data),np.asarray(a)

def main():
  # generate_supervised_training_data(100000)
  # obtain training data
  s_data,a = load_supervised_training_data('./train_data')
  rl_agent = ReinforcementLearningAgent(144,8)
  rl_agent.update_supervised_policy(s_data,a)
  episode = generate_custom_policy_game(rl_agent,rl_agent)

if __name__ == '__main__':
    main()
