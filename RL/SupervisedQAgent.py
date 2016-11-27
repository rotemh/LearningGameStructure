from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop
if K.backend() == 'tensorflow':
  from tensorflow import stop_gradient
elif K.backend() == 'theano':
  from theano.gradient import disconnected_grad

import tensorflow as tf
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import random
import os 
import scipy.io as sio

class SupervisedQAgent:
  """
  Encodes a supervised Q value network.
  To train the network, call .train(...)
  To predict an action, call .predict_action(s)
  """


  def __init__(self,img_shape,num_actions, training_data_path=None, future_discount=1, simple_model=False,preload_data=False):
    #self.img_size = img_size # length of one of the sides of each image (which is square)
    self.img_shape = img_shape # tuple describing the length and width (in pixels), and number of channels of the image
    self.num_actions = num_actions # overall number of actions one could apply
    self.future_discount = future_discount # amount by which to discount each future action, often written as gamma
    self.simple_model = simple_model
    self.preload_data = preload_data

    self.create_cost_function() # constructs DQN value network and cost function which will then be minimized
    
    # The following is used to initialize episode retrieval
    if training_data_path is not None:
      self.training_data_path = training_data_path
    else:
      self.training_data_path = 'train_data/'

    if self.preload_data:
      self.load_all_data()


  def create_simple_model(self):
    """
    Constructs a low-complexity model to see whether the network learns anything
    """
    dense_init = 'glorot_normal'
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
    id_input = Input( shape=(1,),name='player_id',dtype='float32') # In order to merge later, must be float32 as well
    
    s_img_flat = Flatten()(s_img)
    sup_network_merge = merge([s_img_flat,id_input],mode='concat')
    V = Dense(self.num_actions,activation='sigmoid',
                            init=dense_init)(sup_network_merge)
    self.Q_network = Model(input =[s_img,id_input],output=V)
    self.Q_network.compile(loss='mse',
                            optimizer='adadelta',
                            metrics =['mse']
                            )
    self.Q_network.summary()
  def create_model(self):
    """
    Constructs the value network, which takes an image and player id as input
    and saves the outputted network in self.Q_network.

    Current structure is just identical to the one used by Beomjoon in RLAgent.
    """
    conv_init = 'lecun_uniform'
    dense_init = 'glorot_normal'
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
    id_input = Input( shape=(1,),name='player_id',dtype='float32') # In order to merge later, must be float32 as well
    kernel_size = 2

    # Convnet moves from image to low-dimensional rep
    sup_network_h0 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same', init=conv_init)(s_img)
    sup_network_h0 = MaxPooling2D(pool_size=(8,8))(sup_network_h0)

    sup_network_h1 = Convolution2D(nb_filter = 64,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h0)
    sup_network_h1 = MaxPooling2D(pool_size=(4,4))(sup_network_h1)

    sup_network_h2 = Convolution2D(nb_filter = 64,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h1)
    sup_network_h2 = MaxPooling2D(pool_size=(2,2))(sup_network_h2)

    sup_network_h2 = Flatten()(sup_network_h1)
  
    # Combine player_id and low-dimensional rep
    sup_network_merge = merge([sup_network_h2,id_input],mode='concat')

    # pass through two dense layers
    sup_network_dense1 = Dense(256, activation='relu',init=dense_init)(sup_network_merge)
    sup_network_dense2 = Dense(self.num_actions,activation='sigmoid',
                            init=dense_init)(sup_network_dense1)
    V = sup_network_dense2
    self.Q_network = Model(input =[s_img,id_input],output=V)
    self.Q_network.compile(loss='mse',
                            optimizer='adadelta',
                            metrics =['mse']
                            )
    self.Q_network.summary()

  def create_cost_function(self):
    """
    Constructs the cost-function to be trained in the DQN
    This cost function will be stored in self.train_fn
    and can be descended on by calling self.train_fn(state, next_state, action, reward, terminal)

    The overall cost being encoded is:
    (current_reward - Q_opponent(next_state, best action) - Q(state, action))^2
    TBC this isn't validated, but I borrowed much of this syntax from online and it should work

    """
    state = Input(shape=self.img_shape, dtype='float32')
    player = Input(shape=(1,),dtype='float32')
    next_state = Input(shape=self.img_shape, dtype='float32')
    action = Input(shape=(1,), dtype='int32')
    reward = Input(shape=(1,), dtype='float32')
    terminal = Input(shape=(1,), dtype='int32') # 0 if not terminal, 1 if yes

     # constructs the value network
    if self.simple_model:
      self.create_simple_model()
    else:
      self.create_model()
    state_value = self.Q_network([state, player]) # Q(s,*)
    # This next bit computes a non-differentiable value of Q(next_s,*)
    # It is non-differentiable because it shouldn't be updated in the derivative
    # However, WE HAVE NOT YET IMPLEMENTED SAVING AN OLD NETWORK AND COMPUTING Q_OLD
    # Not sure this is necessary for supervised learning only
    next_player = 1 - player
    if K.backend() == 'tensorflow':
      next_state_value = stop_gradient(self.Q_network([next_state, next_player]))
    elif K.backend() == 'theano':
      next_state_value = disconnected_grad(self.Q_network([next_state, next_player]))
    else:
      raise ValueError("Must have one of these two backends, tensorflow or theano")

    # The below line calculates the optimal value starting from Q(next_s,*)
    # Note that this assumes the next state is the opponent's state, so it takes the opposite of that value
    future_value = (1-tf.to_float(terminal)) * (-1)*K.max(next_state_value,axis=1, keepdims=True) # 0 if terminal, otherwise best next move
    discounted_future_value = self.future_discount*future_value # we aren't actually doing discounting, but this would discount future reward
    target = reward + discounted_future_value # this is what we want our Q value to add up to
    action_mask = K.equal(tf.reshape(tf.constant(np.arange(self.num_actions),dtype='int32'),[1, -1]), tf.reshape(action,[-1, 1]))
    action_indexed_state_value = tf.reshape(K.sum(state_value * tf.to_float(action_mask), axis=1),[-1,1])
    cost = K.mean(((action_indexed_state_value - target)**2)) # take the MSE of our current value w.r.t. it
    opt = RMSprop(.0001)
    params = self.Q_network.trainable_weights
    updates = opt.get_updates(params, [], cost) # instantiates optimizer
    # the last line creates a callable function to run the optimizer for a given batch specified by those 5 arguments
    self.cost_fn = K.function([state, next_state, action, reward, terminal, player], [cost])
    self.train_fn = K.function([state, next_state, action, reward, terminal, player], [cost], updates=updates)

  def train(self, num_batches=5000, minibatch_size=128):
    """
    Trains the SupervisedQAgent using episodes retrieved from its encoded directory
    Currently contains a bunch of hyperparameters, we can make them tweakable if we like
    """
    SAVE_FREQUENCY=100 # how often to save the weights
    current_cost = np.inf
    for i in xrange(num_batches):
      current_cost = self.update_batch(minibatch_size) # repeatedly update the network in batches
      if i%SAVE_FREQUENCY == 0:
        print("Updating batch %d, current error: %f")%(i,self.compute_error())
        self.Q_network.save_weights('./qWeights/sup/supweights.h5')
    self.Q_network.save_weights('./qWeights/sup/supweights.h5')


  def compute_error(self, num_validation_episodes=256):
    """
    Computes the error associated with a random subset
    of num_validation_episodes episodes.
    """
    total_cost = 0

    for j in xrange(num_validation_episodes):
      episode = self.get_random_episode()
      state, next_state, action, reward, terminal, player = episode
      total_cost += self.cost_fn([state,next_state,action,reward,terminal,player])[0]

    return total_cost


  def update_batch(self, minibatch_size=128):
    """
    Updates the self.Q_network with a single minibatch of frames
    Each frame goes from state to new_state, using action "action", with reward "reward"
    If the state is terminal, "terminal" is True

    These transitions are drawn in random order from a set of episodes 
    here we draw minibatch_size%8 episodes at random

    each episode - [state, next_state, action, reward, terminal]
    """
    # Create empty containers for the tuples we'll train on
    state = np.zeros((minibatch_size,) + self.img_shape)
    new_state = np.zeros((minibatch_size,) + self.img_shape)
    action = np.zeros((minibatch_size, 1), dtype=np.int32)
    reward = np.zeros((minibatch_size, 1), dtype=np.float32)
    terminal = np.zeros((minibatch_size, 1), dtype=np.int32)
    player = np.zeros((minibatch_size, 1), dtype=np.float32)

    # Retrieve a bunch of episodes to extract transitions from them
    num_episodes_to_retrieve = (np.floor(minibatch_size/8.)).astype(int) # on average, 8 transitions from each episode

    episodes = []
    for i in xrange(num_episodes_to_retrieve):
      episode = self.get_random_episode()
      episodes.append(episode)

    # Now retrieve random samples from the different episodes
    for i in xrange(minibatch_size):
      episode = random.choice(episodes) # pick a random episode
      #note that we pick our frames from the same random subset of pre-cached episodes
      #because uploading a new episode for every frame would take a long long time
      frame = random.choice(episode) # pick a random moment in the episode
      s, ns, a, r, t, p = frame
      # add the frame to the tuples to be trained on
      state[i] = s
      new_state[i] = ns
      action[i] = a
      reward[i] = r
      terminal[i] = t
      player[i] = p

    # train on the frames we just extracted
    # NOTE: there might be a better way to train on a custom function?
    cost_list = self.train_fn([state, new_state, action, reward, terminal, player])
    cost = cost_list[0] # weird quirk of keras requires us to return cost as single-element list
    return cost

  def get_random_episode(self):
    """
    gets a given episode, en entire run-though of one game.

    The episode is returned as a list of transitions, where each
    transition is of the form:
       [state, next_state, action, reward, terminal]
    """
    if self.preload_data:
      return random.choice(self.all_episodes)

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

    episode = matdata_to_npdata(data)

    return episode

  def load_all_data(self):
    """
    Loads all episodes in the training directory into self.all_episodes
    For each episode, stores its relevant variables as described in "matdata_to_npdata"
    """
    self.all_episodes = []
    train_files = os.listdir(self.training_data_path)
    
    for game in train_files:
      try:
        data = sio.loadmat( self.training_data_path +'/'+game )['train_data']
        episode = matdata_to_npdata(data)
        self.all_episodes.append(episode)
      except:
        # corrupted file
        continue
    print len(self.all_episodes)

  def matdata_to_npdata(data):
    """
    Converts mat representation of data to a tuple numpy arrays
    of the following data:
    [state, next_state, action, reward, terminal, player]
    """
    n = data.shape[0]
    state = np.zeros((n,) + self.img_shape)
    next_state = np.zeros((n,) + self.img_shape)
    action = np.zeros((n, 1), dtype=np.int32)
    reward = np.zeros((n, 1), dtype=np.float32)
    terminal = np.zeros((n, 1), dtype=np.int32)
    player = np.zeros((n, 1), dtype=np.float32)

    for i in xrange(data.shape[0]):
      state[i] = data[i][1]
      action[i] = data[i][2][0][0][1][0][0] # col
      next_state[i] = data[i][3]
      player[i] = data[i][0]
      reward[i] = data[i][4][0][player]
      terminal[i] = 1 if reward != 0 else 0
    episode = [state,next_state,action,reward,terminal,player]
    return episode

  def predict_Q_value(self,s, player):
    return self.Q_network.predict([s, np.asarray(player)])

  def predict_action(self,s, player):
    return np.argmax(self.Q_network.predict([s, np.asarray(player)]))

