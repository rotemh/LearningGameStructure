#!/usr/bin/python
from MCTS.sim import *
import numpy as np
import scipy.io as sio
import os
import sys
import threading
from Queue import Queue
from threading import Thread



q = Queue(maxsize = 0)
def worker(q):
	while True:
		episode_number = q.get()
		print "Creating epsiode number %s" % str(episode_number)
		generate_supervised_training_data(episode_number)
		q.task_done()
		

def generate_supervised_training_data(episode_num, time_limit=0.5, file_path='dataset/'):

	train_data= []
	episode = generate_uct_game(time_limit)
	win_player_id = np.argmax(episode[-1][-1])
	winner_train_data = [e for e in episode if e[0] == win_player_id]
	loser_train_data = [e for e in episode if e[0] != win_player_id]
	sio.savemat(file_path + str(episode_num)+'.mat',{'winner_train_data':winner_train_data,'loser_train_data':loser_train_data,\
								'uct_time_limit':time_limit})
	return
	
def main():
	if len(sys.argv) < 2:
		raise Exception("Syntax: python %s episode_numbers" % (sys.argv[0]))
	else:
		num_of_episodes = int(sys.argv[1])
		
	file_path = 'dataset'
	if not os.path.isdir(file_path):
		os.makedirs(file_path)	
	
	for t in xrange(8):
		t = Thread(target=worker, args=(q,))
		t.setDaemon(True)
		t.start()
		
	for i in xrange(num_of_episodes):
		q.put(i)
	
	q.join()



if __name__ == '__main__':
		main()
