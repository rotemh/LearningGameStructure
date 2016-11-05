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
	train_files = os.list_dir(train_dir)
	for f in train_files:
		asdf = sio.loadmat( f )
		import pdb;pdb.set_trace()

def main():
  rl_agent = ReinforcementLearningAgent(128)

  # obtain training data
  #generate_supervised_training_data(100000)
	#train_data = load_supervised_training_data('./train_data')

if __name__ == '__main__':
    main()
