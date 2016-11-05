from MCTS.sim import *
from RL.RLAgent import *
import numpy as np
import scipy.io as sio

def generate_training_data(num_episodes):
  train_data= []
  for x in xrange(num_episodes): 
    print x
    episode = generate_uct_game(time_limit=0.5)
    win_player_id = np.argmax(episode[-1][-1])
    train_data = [e for e in episode if e[0] == win_player_id]
    sio.savemat('/media/beomjoon/My Passport/vision_project/supervised_data/train_data' + str(x)+'.mat',{'train_data':train_data})

def main():
  rl_agent = ReinforcementLearningAgent(128)

  # obtain training data
  generate_training_data(100000)
  import pdb;pdb.set_trace()

  

if __name__ == '__main__':
    main()
