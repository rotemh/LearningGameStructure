from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

class ReinforcementLearningAgent:
  def __init__(self,img_size,num_actions):
    s_img = Input( shape=(img_size,img_size,3),name='s_img',dtype='float32')
    self.num_actions  = num_actions
    kernel_size = 2

    sup_network_h0 = Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size, border_mode='same')(s_img)
    sup_network_h0 = MaxPooling2D(pool_size=(2,2))(sup_network_h0)
    sup_network_h1 = Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size, border_mode='same')(sup_network_h0)
    sup_network_h0 = MaxPooling2D(pool_size=(2,2))(sup_network_h1)
    sup_network_h1 = Flatten()(sup_network_h1)
  
    sup_network_a1 = Dense(num_actions,activation='softmax')(sup_network_h1) # different output layers for each action
    sup_network_a2 = Dense(num_actions,activation='softmax')(sup_network_h1)
    V = merge([sup_network_a1,sup_network_a2],mode='concat')
    self.sup_policy = Model(input =s_img,output=V)
    self.sup_policy.compile(loss='categorical_crossentropy',optimizer='adadelta')
  
    #TODO: implement the rest
    # define the value network
    self.value_network = Sequential() 
    self.value_network.add( Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size, input_shape=(img_size,img_size,3)) )
    self.value_network.add( Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size ))
    self.value_network.add( Convolution2D(nb_filter = 32,nb_row=kernel_size,nb_col=kernel_size))
    self.value_network.add(Flatten())
    self.value_network.add( Dense(1,activation='linear') )
    self.value_network.compile(loss = 'mse',optimizer='adam')

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

  def update_supervised_policy(self,state,a1,a2):
    a1 = np_utils.to_categorical(a1, self.num_actions).astype('int32')
    a2 = np_utils.to_categorical(a2, self.num_actions).astype('int32')
    actions = np.c_[a1,a2]
    state = np.asarray(state)
    self.sup_policy.fit(state,actions)
  
  def update_value_network(self,s,v):
    self.value_network.fit( s,v,nb_epoch=100 )

  def predict_value(self,s):
    return self.value_network.predict(s)

  def predict_action(self,s,policy='supervised'):
    s = np.asarray(s)

    if len(np.shape(s)) == 3:
      s = s.reshape((1,np.shape(s)[0],np.shape(s)[1],np.shape(s)[2]))

    if policy=='supervised':
        action_prob = self.sup_policy.predict(s)
        col_action_prob = action_prob[0,0:8]
        row_action_prob = action_prob[0,8:]
        col_action = np.argmax(col_action_prob)
        row_action = np.argmax(row_action_prob)
        return col_action,row_action
    else:
        return self.policy_network.predict(s)
    

  def q_learning_value_network(self,s,sprime,a,r):
    #NOTE: not sure if we need this
    pass
  
