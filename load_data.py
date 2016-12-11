import os
import pickle
import numpy as np
import scipy.io as sio

def yield_sup_policy_data( train_dir):
  train_files = os.listdir(train_dir)
  
  s_data = []
  a_data = []
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
        s_data.append(data[i]['s_img'])
        a_data.append(data[i]['action']) # col stored only
        """
        player_id.append(data[i]['player_id'])
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
      """
      if len(a_data) >= 5000:
        yield np.asarray(s_data),np.asarray(a_data)
        s_data=[]
        a_data=[]
        """
        yield np.asarray(s_data),np.asarray(a),\
          np.asarray(player_id),np.asarray(reward),np.asarray(value),\
          np.asarray(s_board_data),np.asarray(sprime_board_data)
        s_data = []
        a = []
        sprime_data = []
        s_board_data = []
        sprime_board_data = []
        reward = []
        player_id = []
        value = []
        print len(a)
        """
    except ValueError:
      print f + ' is corrupted'
