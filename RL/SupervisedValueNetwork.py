from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from RL.LossGrapher import LossGrapher

import pickle
import os
import numpy as np

class SupervisedValueNetworkAgent:
  def __init__(self,img_shape):
    self.img_shape = img_shape 
    self.create_supervised_v_network_model()

  def create_supervised_v_network_model(self):
    conv_init = 'lecun_uniform'
    dense_init = 'glorot_normal'
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
    #id_input = Input( shape=(1,),name='player_id',dtype='float32')
    kernel_size = 4

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
    sup_network_h2 = Flatten()(sup_network_h3)

    sup_network_merge = sup_network_h2 
    sup_network_batch_normed = BatchNormalization()(sup_network_merge)
    sup_network_v = Dense(1,activation='linear',
                            init=dense_init)(sup_network_merge)
    V = sup_network_v
    self.sup_v_network = Model(input =s_img,output=V)
    self.sup_v_network.compile(loss='mse',
                            optimizer='adadelta',
                            metrics =['accuracy']
                            )
    self.h0_output = Model(input=[s_img],output = sup_network_h0)
    self.h1_output = Model(input=[s_img],output = sup_network_h1)
    self.h2_output = Model(input=[s_img],output = sup_network_h2)
    self.h3_output = Model(input=[s_img],output = sup_network_h3)

  def get_intermediate_layer_outputs(self,s):
    s = (np.asarray(s).copy()).astype('float32')
    s = self.datagen.standardize(s)
    return self.h0_output.predict(s),self.h1_output.predict(s),\
            self.h2_output.predict(s),self.h3_output.predict(s),\

  def update_v_network(self,state,v):
    state =state.astype('float32')
    self.datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True)
    self.datagen.fit(state)
    pickle.dump( self.datagen,open('./valueNetworkWeights/sup/datagen.p','wb') )

    state = np.asarray(state)
    early = EarlyStopping(monitor='val_loss', patience=20000, verbose=0, mode='auto')
    state = self.datagen.standardize(state)
    checkpoint = ModelCheckpoint(filepath=\
                          './valueNetworkWeights/sup/sup_weights.{epoch:02d}-{val_acc:.5f}.hdf5',\
                          monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    loss_graph = LossGrapher()
    self.sup_v_network.fit(state,v,nb_epoch=10000,
                        callbacks=[early,checkpoint,loss_graph],
                        batch_size = 32,
                        validation_split = 0.1)

  def load_train_results(self):
    """
    self.sup_v_network.load_weights('./valueNetworkWeights/sup/old_good/sup_weights.04-0.92093.hdf5')
    self.datagen = pickle.load( open( "./valueNetworkWeights/sup/old_good/datagen.p" ) )
    """
    self.sup_v_network.load_weights('./valueNetworkWeights/sup/sup_weights.17-0.91537.hdf5')
    self.datagen = pickle.load( open( "./valueNetworkWeights/sup/datagen.p" ) )
    mock_s = np.zeros((1,144,144,3))
    self.predict_value(mock_s)
  
  def predict_value(self,s):
    s = (np.asarray(s).copy()).astype('float32')
    s = self.datagen.standardize(s)
    if len(np.shape(s)) == 3:
      s = s.reshape((1,np.shape(s)[0],np.shape(s)[1],np.shape(s)[2]))

    v= self.sup_v_network.predict(s)
    print v
    return v
    
  
