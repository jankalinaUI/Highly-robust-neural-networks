# Creating Quantile RBF netowrk class
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras import regularizers
from tensorflow.keras import layers
from keras.models import Sequential
from keras.engine.input_layer import Input
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2
from sklearn.model_selection import GridSearchCV
#from RBF_tf import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
#from Losses import psi,quantile_nonlinear,least_weighted_square

class QuantileNetwork:
  
  """
  Quantile Network class can be used for define models for inter quantile
  training, but also for creating basic models
  
  Parameters:
  
  thau, betas, units, input_shape,x,neurons1,neurons2
  
  """
	
  def __init__(self,thau,betas, units, input_shape,x, neurons1,neurons2):
		
    self.thau = thau
    self.betas = betas
    self.units = units
    self.shape = input_shape
    self.x = x
    self.neurons1 = neurons1
    self.neurons2 = neurons2
		
  #MLP_model
  def MLP_model(self,neurons1 = None,neurons2= None,input_shape = None, loss = None):
        
    self.loss =  'mean_squared_error' if loss is None  else loss
    self.input_shape =  self.shape if input_shape is None else input_shape
    self.neurons1 = 128 if neurons1 is None else neurons1
    self.neurons2 = 64 if neurons2 is None else neurons2
		
    
    inputs = Input(shape = (self.input_shape,))
    layer = Dense(self.neurons1, activation = K.sigmoid)(inputs)
    lay = Dense(self.neurons2,activation = K.sigmoid)(layer)
    out = Dense(1)(lay)
    

    model = Model(inputs = inputs , outputs = out)

    model.compile(loss = self.loss, optimizer = RMSprop())
       
    return model

  #RBF model		
  def RBF_model(self,x, input_shape, units, betas, loss):
    
    self.loss =  'mean_squared_error' if loss is None else loss
    self.input_shape = self.shape if input_shape is None else  input_shape
    self.units =  40 if loss is None else  units
    self.betas = self.betas if betas is None else betas
    self.X =  self.x if x is None else  x
    
	
    inputs = Input(shape = (self.input_shape,))
    rbflayer = RBFLayer(output_dim = self.units,
                        betas=self.betas,
                        initializer = InitCentersRandom(self.X))
    
    rbf = rbflayer(inputs)
    out = Dense(1)(rbf)
      
    model = Model(inputs = inputs , outputs = out)
    model.compile(loss = self.loss,
                  optimizer = RMSprop())
        
    return model
		
  def QMLP(self):
  
    thau_upper = self.thau
    thau_lower = 1-self.thau
	
    model_u = self.MLP_model(input_shape = self.shape,
							 loss = quantile_nonlinear(thau_upper),
                             neurons1 = self.neurons1,
                             neurons2 = self.neurons2)
    model_l = self.MLP_model(input_shape = self.shape,
						     loss = quantile_nonlinear(thau_lower),
                             neurons1 = self.neurons1,
                             neurons2 = self.neurons2)

    return model_u,model_l
	
  def QRBF(self):
  
    thau_upper = self.thau
    thau_lower = 1-self.thau

	
  
    model_u = self.RBF_model(x = self.x, 
							input_shape = self.shape,
							betas = self.betas, 
							units = self.units, 
							loss = quantile_nonlinear(thau_upper))
    model_l = self.RBF_model(x = self.x, 
							input_shape = self.shape,
							betas = self.betas, 
							units = self.units, 
							loss = quantile_nonlinear(thau_lower))

    return model_u,model_l
  
  def __repr__(self):
    return '[arguments: thau - %s, betas - %s, shape - %s, units - %s]' % (self.thau,self.betas,self.shape,self.units)