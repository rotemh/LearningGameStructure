from MCTS.sim import *
from RL.RLAgent import *
import numpy as np
import scipy.io as sio
import os

def generate_supervised_training_data(num_episodes):
  train_data= []
  for x in xrange(num_episodes): 
    print x
    episode = generate_uct_game(time_limit=0.5)
    win_player_id = np.argmax(episode[-1][-1])
    train_data = [e for e in episode if e[0] == win_player_id]
    sio.savemat('/media/beomjoon/My Passport/vision_project/supervised_data/train_data' + str(x)+'.mat',{'train_data':train_data})

def load_supervised_training_data( train_dir ):
  train_files = os.listdir(train_dir)
  
  s_data = []
  a1 = []; a2=[]
  sprime_data = []
  reward = []
  for f in train_files:
    try:
      data = sio.loadmat( train_dir+'/'+f )['train_data']
      s_data += [data[0][1]]
      a1 += [data[0][2][0][0][1][0][0]]
      a2 += [data[0][2][0][0][2][0][0]]
      sprime_data += [data[0][3]]
      reward += [data[0][4]]
    except:
      # some files corrupted
      continue
  return s_data,a1,a2

def main():

  # obtain training data
  #generate_supervised_training_data(100000)
  s_data,a1,a2 = load_supervised_training_data('./train_data')
  print np.shape(s_data)
  rl_agent = ReinforcementLearningAgent(144,8)
  rl_agent.update_supervised_policy(s_data,a1,a2)
  episode = generate_custom_policy_game(time_limit=0.5)

if __name__ == '__main__':
    main()
