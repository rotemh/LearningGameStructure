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
    self.h0_output = Model(input=[s_img],output = sup_network_h0)
    self.h1_output = Model(input=[s_img],output = sup_network_h1)
    self.h2_output = Model(input=[s_img],output = sup_network_h2)
    self.h3_output = Model(input=[s_img],output = sup_network_h3)
  

  def get_intermediate_layer_outputs(self,s):
    s = (np.asarray(s).copy()).astype('float32')
    s = self.datagen.standardize(s)
    return self.h0_output.predict(s),self.h1_output.predict(s),\
            self.h2_output.predict(s),self.h3_output.predict(s)

  def process_data(self,d):
    s = d[0]
    s = s.astype('float32')
    state = self.datagen.standardize(s)
    a = d[1]
    a = a.astype('float32')
    action = np_utils.to_categorical(a, self.num_actions).astype('int32')
    return state,action
  def update_supervised_policy(self,yield_sup_policy_data):
    """
    state =state.astype('float32')
    a = a.astype('float32')
    self.datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True)
    self.datagen.fit(state)
    pickle.dump( self.datagen,open('./policyWeights/sup/datagen.p','wb') )
    action = np_utils.to_categorical(a, self.num_actions).astype('int32')
    state = np.asarray(state)
    state = self.datagen.standardize(state)
    """
    self.datagen = pickle.load( open( "./policyWeights/sup/datagen.p", "rb" ) )
    early = EarlyStopping(monitor='val_loss', patience=20000, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath=\
                                './policyWeights/sup/sup_weights.{epoch:02d}-{val_acc:.5f}.hdf5',\
                                  monitor='val_acc', verbose=0,  mode='auto')
    loss_graph = LossGrapher()
    nb_epoch = 10000
    counter = 0
    for e in range(nb_epoch):
      data_gen = yield_sup_policy_data('/home/beomjoon/LearningGameStructure/dataset/')
      for d in data_gen:
        state,action = self.process_data(d)
        self.sup_policy.fit([state],action,nb_epoch=1,batch_size = 32,verbose=True,validation_split=0.1)
        weight_f = './policyWeights/sup/sup_weights.e.'+str(e)+'.counter.'+str(counter)+'.hdf5'
        self.sup_policy.save(weight_f)
        counter += 1
        print counter
        if counter % 5000 ==0:
          test_d = data_gen.next()
          state,action = self.process_data(test_d)
          pred = self.sup_policy.predict(state)
          val_acc = np.sum(np.argmax(pred,axis=1) == np.argmax(action,axis=1))/float(len(pred))
          weight_f = './policyWeights/sup/sup_weights.{epoch:'+ str(e) + '}-{val_acc:' + str(val_acc) +'}.hdf5'
          self.sup_policy.save(weight_f)
      print 'epoch = ' + str(e) + ' '

  def load_train_results(self):
    self.sup_policy.load_weights('./policyWeights/sup/sup_weights.22-0.38204.hdf5')
#    self.datagen = pickle.load( open( "./policyWeights/sup/datagen_0.38.p", "rb" ) )
    self.datagen = pickle.load( open( './policyWeights/sup/datagen_0.38.p', "rb" ) )
    mock_s = np.zeros((1,144,144,3))
    self.predict_action(mock_s)
  
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
    print action_prob
    if np.shape(s)[0] == 1:
      return action_prob[0]
    else:
      return action_prob
