from MCTS.sim import *
#from RL.RLAgent import *
from MCTS.game import * 
from RL.SupervisedPolicy import SupervisedPolicyAgent
import numpy as np
import scipy.io as sio
import os
import sys
import threading
from tester import test_policy_vs_MCTS
from Queue import Queue
from threading import Thread
import pickle 
from train_data_generate_functions import *

q = Queue(maxsize = 0)
def worker_p(q):
  while True:
    episode_number = q.get()
    print "Creating epsiode number %s" % str(episode_number)
    generate_policy_training_data(episode_number)
    q.task_done()

def worker_v(q):
  while True:
    episode_number = q.get()
    print "Creating epsiode number %s" % str(episode_number)
    generate_v_training_data(episode_number)
    q.task_done()

def main():
  if len(sys.argv) < 2:
    raise Exception("Syntax: python %s episode_numbers" % (sys.argv[0]))
  else:
    num_of_episodes = int(sys.argv[1])
    train_data_type = sys.argv[2]

  if train_data_type =='v':
    file_path = 'v_dataset'
  elif train_data_type == 'p':
    file_path = 'dataset'
  else:
    print "Wrong train data type"
    return
  if not os.path.isdir(file_path):
    os.makedirs(file_path)  
  
  if train_data_type == 'p':
    for t in xrange(8):
      t = Thread(target=worker_p, args=(q,))
      t.setDaemon(True)
      t.start()
      
    for i in xrange(num_of_episodes):
      q.put(i)
    
    q.join()
  else:
    print 'generating value network data'
    for t in xrange(8):
      t = Thread(target=worker_v, args=(q,))
      t.setDaemon(True)
      t.start()
      
    for i in xrange(313,num_of_episodes):
      q.put(i)
    
    q.join()


if __name__ == '__main__':
  main()
  #generate_supervised_training_data(1)
