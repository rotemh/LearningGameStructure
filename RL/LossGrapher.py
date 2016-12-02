from keras.models import Sequential ,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class LossGrapher(Callback):
  def __init__(self):
    return

  def on_train_begin(self,logs={}):
    self.acc = []
    self.val_acc = []
    self.loss = []
    self.val_loss = []

  def on_epoch_end(self,epoch,logs={}):
    self.loss.append(logs.get('loss'))
    self.val_loss.append(logs.get('val_loss'))
    self.acc.append(logs.get('acc'))
    self.val_acc.append(logs.get('val_acc'))
    if epoch % 10 == 0:
      self.plot_history(epoch)
  
  def plot_history(self,epoch):
    # summarize history for accuracy    
    fig = plt.figure()
    plt.plot(self.acc)
    plt.plot(self.val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./lossPlot/model_accuracy.png',dpi=fig.dpi)

    # summarize history for loss
    fig = plt.figure()
    plt.plot(self.loss)
    plt.plot(self.val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./lossPlot/model_loss_epoch.png',dpi=fig.dpi)
