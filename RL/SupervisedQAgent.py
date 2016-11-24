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

class SupervisedQAgent:
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
    print(self.get_random_episode() )#Test

  def create_Q_model(self):
    conv_init = 'lecun_uniform'
    dense_init = 'glorot_normal'
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
    id_input = Input( shape=(1,),name='player_id',dtype='float32')
    kernel_size = 2

    sup_network_h0 = Convolution2D(nb_filter = 16,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same', init=conv_init)(s_img)
    sup_network_h0 = MaxPooling2D(pool_size=(2,2))(sup_network_h0)

    sup_network_h1 = Convolution2D(nb_filter = 16,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h0)
    sup_network_h1 = MaxPooling2D(pool_size=(2,2))(sup_network_h1)

    sup_network_h2 = Convolution2D(nb_filter = 16,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h1)
    sup_network_h2 = MaxPooling2D(pool_size=(2,2))(sup_network_h2)

    sup_network_h2 = Flatten()(sup_network_h1)
  
    sup_network_merge = merge([sup_network_h2,id_input],mode='concat')
    sup_network_a = Dense(self.num_actions,activation='softmax',
                            init=dense_init)(sup_network_merge)
    V = sup_network_a
    self.Q_network = Model(input =[s_img,id_input],output=V)
    self.Q_network.compile(loss='mse',
                            optimizer='adadelta',
                            metrics =['mse']
                            )
    self.Q_network.summary()

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
    Updates the self.Q_network with a single minibatch of frames
    Each frame goes from state to new_state, using action "action", with reward "reward"
    If the state is terminal, "terminal" is True

    These transitions are drawn in random order from a set of episodes 
    here we draw minibatch_size%8 episodes at random

    each episode - [state, next_state, action, reward, terminal]
    """
    state = numpy.zeros((self.mbsz,) + self.state_size)
    new_state = numpy.zeros((self.mbsz,) + self.state_size)
    action = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
    reward = numpy.zeros((self.mbsz, 1), dtype=numpy.float32)
    terminal = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)

    # Retrieve a bunch of episodes to extract transitions from them
    num_episodes_to_retrieve = minibatch_size%8 # on average, 8 transitions from aech episode

    episodes = []
    for i in xrange(num_episodes_to_retrieve):
      episode = self.get_random_episode()
      episodes.append(episode)

    # Now retrieve random samples from the different episodes
    for i in xrange(minibatch_size):
      episode = random.randint(0,len(episodes)-1)
      s, ns, a, r, t = episodes[episode]
      frame = random.randint(0,len(s)-1)
      state[i] = s[frame]
      new_state[i] = ns[frame]
      action[i] = a[frame]
      reward[i] = r[frame]
      terminal[i] = t[frame]

    cost = self.train_Q_fn(state, new_state, action, reward, terminal)
    return cost

  def get_random_episode(self):
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
