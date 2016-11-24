from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop
if K.backend == 'tensorflow':
  from tensorflow import stop_gradient
elif K.backend == 'theano':
  from theano.gradient import disconnected_grad
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import random
import os 
import scipy.io as sio

class ReinforcementLearningAgent:
  def __init__(self,img_shape,num_actions, training_data_path=None, future_discount=1):
    #self.img_size = img_size # length of one of the sides of each image (which is square)
    self.img_shape = img_shape # tuple describing the length and width (in pixels), and number of channels of the image
    self.num_actions = num_actions # overall number of actions one could apply
    self.future_discount = future_discount

    self.create_supervised_Q_cost_function()
    
    if training_data_path is not None:
      self.training_data_path = training_data_path
    else:
      self.training_data_path = '../train_data/'

    print("test")
    print(self.getRandomEpisode() )#Test

  def create_Q_model(self):
    self.Q_network = Sequential()
    kernel_size = 2
    self.Q_network.add( Convolution2D(nb_filter = 16,nb_row=kernel_size,nb_col=kernel_size,\
     input_shape=self.image_shape, subsample=(4,4), activation='relu', input_dim=self.img_shape) )
    self.Q_network.add( Convolution2D(nb_filter = 16,nb_row=kernel_size,nb_col=kernel_size,\
     input_shape=self.image_shape, subsample=(4,4), activation='relu') )
    self.Q_network.add( Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size,\
     subsample=(2,2), activation='relu') )
    self.Q_network.add(Flatten())
    self.Q_network.add( Dense(256,activation='relu'))
    self.Q_network.add( Dense(self.num_actions))
    self.Q_network.compile(loss = 'mse',optimizer='adam')

  def create_supervised_Q_cost_function(self):
    state = Input(shape=self.img_shape, dtype='float32')
    next_state = Input(shape=self.img_shape, dtype='float32')
    action = Input(shape=(1,), dtype='int32')
    reward = Input(shape=(1,), dtype='float32')
    terminal = Input(shape=(1,), dtype='int32') # 0 if not terminal, 1 if yes

    self.create_Q_model()
    state_value = self.Q_network(state)
    if K.backend == 'tensorflow':
      next_state_value = stop_gradient(self.Q_network(next_state))
    elif K.backend == 'theano':
      next_state_value = disconnected_grad(self.Q_network(next_state))
    else:
      raise IllegalArgumentException("Must have one of these two backends, tensorflow or theano")

    future_value = (1-terminal) * next_state_value.max(axis=1, keepdims=True) # 0 if terminal, otherwise best next move
    discounted_future_value = self.future_discount*future_value
    target = reward + discounted_future_value
    cost = ((state_value[:,action] - target)**2).mean()
    opt = RMSprop(.0001)
    params = self.Q_network.trainable_weights
    updates = opt.get_updates(params, [], cost)
    self.train_Q_fn = K.function([state, next_state, action, reward, terminal], cost, updates=updates)

  def train_supervised_Q(self, num_batches=1000, minibatch_size=32):
    SAVE_FREQUENCY=100
    current_cost = np.inf
    for i in xrange(num_batches):
      print("Updating batch %d, current error: %f")%(i,current_cost)
      current_cost = self.update_batch_supervised_Q()
      if i%SAVE_FREQUENCY == 0:
        self.Q_network.save_weights('./qWeights/sup/supweights.h5')
    self.Q_network.save_weights('./qWeights/sup/supweights.h5')

  def update_batch_supervised_Q(self, minibatch_size=32):
    """
    Updates the self.Q_network

    each episode - [state, next_state, action, reward, terminal]
    """
    state = numpy.zeros((self.mbsz,) + self.state_size)
    new_state = numpy.zeros((self.mbsz,) + self.state_size)
    action = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
    reward = numpy.zeros((self.mbsz, 1), dtype=numpy.float32)
    terminal = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
    for i in xrange(minibatch_size):
      # TODO: get this from a random episode
      s, ns, a, r, t = self.get_random_episode()
      state[i] = s
      new_state[i] = ns
      action[i] = a
      reward[i] = r
      terminal[i] = t

    cost = self.train_Q_fn(state, new_state, action, reward, terminal)
    return cost

  def getRandomEpisode(self):
    """
    gets a given episode
    """
    train_files = os.listdir(self.training_data_path)
    game = random.choice(train_files)
    gotData = False

    while not gotData:
      game = random.choice(train_files)
      try:
        data = sio.loadmat( self.training_data_path +'/'+game )['train_data']
        gotData = True
      except:
        # corrupted file
        continue

    i = random.randint(0, data.shape[0]-1)
    player = data[i][0]
    state = data[i][1]
    action = data[i][2][0][0][1][0][0] # col
    next_state = data[i][3]
    reward = data[i][4][0][player]
    print(reward)
    terminal = 1 if reward != 0 else 0
    gotData = True

    return [state, next_state, action, reward, terminal]

  def predict_Q_value(self,s):
    return self.Q_network.predict(s)

  def predict_action(self,s):
    return np.argmax(self.Q_network.predict(s))


  def q_learning_Q_network(self,s,sprime,a,r):
    #NOTE: not sure if we need this
    pass


#a = ReinforcementLearningAgent((144,144,3),8)  
