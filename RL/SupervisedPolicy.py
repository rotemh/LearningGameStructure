from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from RL.LossGrapher import LossGrapher

import os
import pickle
import numpy as np

class SupervisedPolicyAgent:
  def __init__(self,img_shape,num_actions):
    #self.img_size = img_size # length of one of the sides of each image (which is square)
    self.img_shape = img_shape # tuple describing the length and width (in pixels), and number of channels of the image
    self.num_actions = num_actions # overall number of actions one could apply
    
    self.create_supervised_policy_model()

  def create_supervised_policy_model(self):
    conv_init = 'lecun_uniform'
    dense_init = 'glorot_normal'
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
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
    """
    sup_network_h3 = Convolution2D(nb_filter = 32,
                                   nb_row=kernel_size,
                                   nb_col=kernel_size, 
                                   border_mode='same',init=dense_init)(sup_network_h3)
    sup_network_h3 = MaxPooling2D(pool_size=(2,2))(sup_network_h3)
    """
    sup_network_h2 = Flatten()(sup_network_h3)
    sup_network_merge = sup_network_h2 #merge([sup_network_h2,id_input],mode='concat')
    
    sup_network_batch_normed = BatchNormalization()(sup_network_merge)
    #TODO: does this make it so that the data is centered? what does batchnormalization actually do?
    sup_network_a = Dense(self.num_actions,activation='softmax',
                            init=dense_init)(sup_network_batch_normed)
    V = sup_network_a
    self.sup_policy = Model(input =s_img,output=V)
    self.sup_policy.compile(loss='categorical_crossentropy',
                            optimizer='adadelta',
                            metrics =['accuracy']
                            )

    # predict intermediate layers
    """
    self.cnn_output = Model(input=[s_img],output = sup_network_h3)
    self.sup_network_merge = Model(input=[s_img],output = sup_network_merge)
    self.sup_network_batch_noremd = Model(input=[s_img],output = sup_network_batch_normed)
    """
  
    self.sup_policy.summary()

  def save_classified_data(self):
    # saves the output of intermediate layers on correctly classified data
    # saves the output of intermediate layers on misclassified data

    # saves the misclassified data
    # saves the correctly classified data
    pass
    
  def update_supervised_policy(self,state,a,player_id):
    state =state.astype('float32')
    a = a.astype('float32')
    player_id = player_id.astype('float32')
    player_id = player_id.reshape((np.shape(player_id)[0], ))
    self.datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True)
    self.datagen.fit(state)
    pickle.dump( self.datagen,open('./policyWeights/sup/datagen.p','wb') )

    action = np_utils.to_categorical(a, self.num_actions).astype('int32')
    state = np.asarray(state)
    early = EarlyStopping(monitor='val_loss', patience=20000, verbose=0, mode='auto')
    state = self.datagen.standardize(state)
    checkpoint = ModelCheckpoint(filepath=\
                                './policyWeights/sup/sup_weights.{epoch:02d}-{val_acc:.5f}.hdf5',\
                                  monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    loss_graph = LossGrapher()
    history = self.sup_policy.fit([state],action,nb_epoch=10000,
                          callbacks=[early,checkpoint,loss_graph],
                          batch_size = 32,
                          validation_split = 0.25)
    self.save_classified_data()

  def load_train_results(self):
    self.sup_policy.load_weights('./policyWeights/sup/sup_weights.369-0.31940.hdf5')
    self.datagen = pickle.load( open( "./policyWeights/sup/datagen.p", "rb" ) )
  
  def predict_action(self,s):
    '''
    This function returns a probability distribution across columns
    when policy is set to supervised. It does nothing when policy
    is not set to supervised since there is no policy_network
    '''
    s = (np.asarray(s).copy()).astype('float32')
    s = self.datagen.standardize(s)

    if len(np.shape(s)) == 3:
      s = s.reshape((1,np.shape(s)[0],np.shape(s)[1],np.shape(s)[2]))
    action_prob = self.sup_policy.predict(s)
    if np.shape(s)[0] == 1:
      return action_prob[0]
    else:
      return action_prob
