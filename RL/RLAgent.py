from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import *
import numpy as np

class ReinforcementLearningAgent:
  def __init__(self,img_shape,num_actions):
    #self.img_size = img_size # length of one of the sides of each image (which is square)
    self.img_shape = img_shape # tuple describing the length and width (in pixels), and number of channels of the image
    self.num_actions = num_actions # overall number of actions one could apply
    
    self.create_supervised_policy_model()

  """
    # define the policy network
    import theano.tensor as T
    def policy_network_loss_func( y, y_pred, base_line ):
      # TODO: something is wrong here; we would like to increase the probability of action that has high advantage value. Comeback later
      return -T.log(y_pred) * base_line 
    policy_network_loss_with_baseline = partial(policy_network_loss_func,base_line= self.value_network)
    policy_network_loss_with_baseline.__name__ = ' policy_network_loss_with_baseline'

    self.policy_network = Sequential()
    self.policy_network.add( Convolution2D(nb_filters = 32,kernel_dim1=kernel_size,kernel_dim2=kernel_size, input_dim=(img_size,img_size,3)) )
    self.policy_network.add( Convolution2D(nb_filters = 32,kernel_dim1=kernel_size,kernel_dim2=kernel_size) )
    self.policy_network.add( Convolution2D(nb_filters = 32,kernel_dim1=kernel_size,kernel_dim2=kernel_size) )
    self.policy_network.add( Dense(num_actions,activation='softmax') )
    self.policy_network.compile(loss = policy_network_with_baseline, optimizer='adam')

  def update_policy_network(s,sprime,a,r):  
    # TODO: make sure you use sprime, a, and r to caculate the gradient; something is wrong here right now
    self.policy_network.fit( s, nb_epoch= 1 ) 
  """

  def create_supervised_policy_model(self):
    s_img = Input( shape=self.img_shape,name='s_img',dtype='float32')
    kernel_size = 2

    sup_network_h0 = Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size, border_mode='same')(s_img)
    sup_network_h0 = MaxPooling2D(pool_size=(2,2))(sup_network_h0)
    sup_network_h1 = Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size, border_mode='same')(sup_network_h0)
    sup_network_h0 = MaxPooling2D(pool_size=(2,2))(sup_network_h1)
    sup_network_h1 = Flatten()(sup_network_h1)
  
    sup_network_a = Dense(self.num_actions,activation='softmax')(sup_network_h1) # different output layers for each action
    V = sup_network_a
    self.sup_policy = Model(input =s_img,output=V)
    self.sup_policy.compile(loss='categorical_crossentropy',optimizer='adadelta')

  def create_supervised_Q_model(self):
    raise NotImplementedError("Doesn't actually work yet")

    state = Input(shape=self.state_size)
    next_state = Input(shape=self.state_size)
    action = Input(shape=(1,), dtype='int32')
    reward = Input(shape=(1,), dtype='float32')
    transition = Input(shape=(1,), dtype='int32')
    self.value_network = Sequential() 
    self.value_network.add( Convolution2D(nb_filter = 16,nb_row=kernel_size,nb_col=kernel_size,\
     input_shape=self.image_shape, subsample=(4,4), activation='relu') )
    self.value_network.add( Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size,\
     subsample=(2,2), activation='relu') )
    self.value_network.add(Flatten())
    self.value_network.add( Dense(256,activation='relu'))
    self.value_network.add( Dense(self.num_actions))
    self.value_network.compile(loss = 'mse',optimizer='adam')

  def update_supervised_policy(self,state,a):
    action = np_utils.to_categorical(a, self.num_actions).astype('int32')
    state = np.asarray(state)
    early = EarlyStopping(monitor='loss', patience=20, verbose=0, mode='auto')
    self.sup_policy.fit(state,action,nb_epoch=100,callbacks=[early])
  
  def update_value_network(self,s,v):
    self.value_network.fit( s,v,nb_epoch=100 )

  def predict_value(self,s):
    return self.value_network.predict(s)

  def predict_action(self,s,policy='supervised'):
    '''
    This function returns a probability distribution across columns
    when policy is set to supervised. It does nothing when policy
    is not set to supervised since there is no policy_network
    '''
    s = np.asarray(s)

    if len(np.shape(s)) == 3:
      s = s.reshape((1,np.shape(s)[0],np.shape(s)[1],np.shape(s)[2]))

    if policy=='supervised':
        action_prob = self.sup_policy.predict(s)
        return action_prob
    else:
        return self.policy_network.predict(s)
    
  def q_learning_value_network(self,s,sprime,a,r):
    #NOTE: not sure if we need this
    pass
  
