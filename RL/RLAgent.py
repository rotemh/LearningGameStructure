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

    self.create_supervised_policy_model()
    
    if training_data_path is not None:
      self.training_data_path = training_data_path
    else:
      self.training_data_path = '../train_data/'

  def create_supervised_policy_model(self):
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
    self.sup_policy = Model(input =[s_img,id_input],output=V)
    self.sup_policy.compile(loss='categorical_crossentropy',
                            optimizer='adadelta',
                            metrics =['accuracy']
                            )
    self.sup_policy.summary()


  def update_supervised_policy(self,state,a,player_id):
    state =state.astype('float32')
    a =a.astype('float32')
    player_id = player_id.astype('float32')
    player_id = player_id.reshape((np.shape(player_id)[0], ))
    self.datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True)
    self.datagen.fit(state)

    action = np_utils.to_categorical(a, self.num_actions).astype('int32')
    state = np.asarray(state)
    early = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    state = self.datagen.standardize(state)
    weight_fname = './policyWeights/sup/supweights.h5'
    if os.path.exists(weight_fname):
      overwrite = bool(int(raw_input('Overwrite sup weights?')))
    else:
      overwrite = True
    if overwrite:
      self.sup_policy.fit([state,player_id],action,nb_epoch=100,
                          callbacks=[early],
                          batch_size = 32,
                          validation_split = 0.1
                          )
      self.sup_policy.save_weights(weight_fname)
    else:
      self.sup_policy.load_weights(weight_fname)

  def predict_action(self,s,policy='supervised'):
    '''
    This function returns a probability distribution across columns
    when policy is set to supervised. It does nothing when policy
    is not set to supervised since there is no policy_network
    '''
    s = np.asarray(s)
    s=self.datagen.standardize(s)

    if len(np.shape(s)) == 3:
      s = s.reshape((1,np.shape(s)[0],np.shape(s)[1],np.shape(s)[2]))

    if policy=='supervised':
        action_prob = self.sup_policy.predict(s)
        return action_prob
    else:
        return self.policy_network.predict(s)



#a = ReinforcementLearningAgent((144,144,3),8)  
