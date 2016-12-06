from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np

class SupervisedValueNetworkAgent:
  def __init__(self,img_shape,num_actions):
    #self.img_size = img_size # length of one of the sides of each image (which is square)
    self.img_shape = img_shape # tuple describing the length and width (in pixels), and number of channels of the image
    self.num_actions = num_actions # overall number of actions one could apply
    self.create_supervised_v_network_model()

  def create_supervised_v_network_model(self):
    conv_init = 'lecun_uniform'
    dense_init = 'glorot_normal'
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
    id_input = Input( shape=(1,),name='player_id',dtype='float32')
    kernel_size = 2

    sup_network_h0 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same', init=conv_init)(s_img)
    sup_network_h0 = MaxPooling2D(pool_size=(2,2))(sup_network_h0)
    sup_network_h1 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h0)
    sup_network_h1 = MaxPooling2D(pool_size=(2,2))(sup_network_h1)
    sup_network_h2 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h1)
    sup_network_h2 = MaxPooling2D(pool_size=(2,2))(sup_network_h2)
    sup_network_h3 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h2)
    sup_network_h3 = MaxPooling2D(pool_size=(2,2))(sup_network_h3)

    sup_network_h3 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h3)
    sup_network_h3 = MaxPooling2D(pool_size=(2,2))(sup_network_h3)
    sup_network_h2 = Flatten()(sup_network_h3)

    sup_network_merge = merge([sup_network_h2],mode='concat')
    sup_network_v = Dense(1,activation='linear',
                            init=dense_init)(sup_network_merge)
    V = sup_network_v
    self.sup_v_network = Model(input =[s_img],output=V)
    self.sup_v_network.compile(loss='mse',
                            optimizer='adadelta',
                            metrics =['accuracy']
                            )
    self.sup_v_network.summary()

  def update_v_network(self,state,player_id,v):
    state =state.astype('float32')
    player_id = player_id.astype('float32')
    player_id = player_id.reshape((np.shape(player_id)[0], ))
    self.datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True)
    self.datagen.fit(state)

    state = np.asarray(state)
    early = EarlyStopping(monitor='val_loss', patience=20000, verbose=0, mode='auto')
    state = self.datagen.standardize(state)
    checkpoint = ModelCheckpoint(filepath=\
                          './valueNetworkWeights/sup/sup_weights.{epoch:02d}-{val_loss:.5f}.hdf5',\
                           monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    self.sup_v_network.fit([state],v,nb_epoch=10000,
                        callbacks=[early,checkpoint],
                        batch_size = 32,
                        validation_split = 0.1
                        )
  
  def predict_value(self,s):
    s = np.asarray(s)
    s = self.datagen.standardize(s)

    if len(np.shape(s)) == 3:
      s = s.reshape((1,np.shape(s)[0],np.shape(s)[1],np.shape(s)[2]))

    return self.sup_v_network.predict(s)
    
  
