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
from rbf import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
from models import QMLP,QRBF,MLP_model,RBF_model

#RBF model
def RBF_model(x, input_shape = 1, units = 40, betas = 2.0, loss = 'mean_squared_error'):
        inputs = Input(shape = (input_shape,))
        rbflayer = RBFLayer(output_dim = units,
                        betas=betas,
                        initializer = InitCentersRandom(x))
        rbf = rbflayer(inputs)
        out = Dense(1)(rbf)
      
        model = Model(inputs = inputs , outputs = out)
        model.compile(loss = loss,
					  optimizer = RMSprop())
        
        return model

#QRBF 
def QRBF(thau, x, input_shape = 1, units = 40, betas = 2.0):
  
  def quantile_nonlinear(y_true,y_pred):
  
    x = y_true - y_pred
    #pretoze sa bude variac tensor, toto je postup pri kerase
  
    return K.maximum(thau * x,(thau - 1) * x)
	
  model = RBF_model(x = x, input_shape = input_shape, betas = betas, units = units, loss = quantile_nonlinear)

  return model
	

#MLP 

def MLP_model(x, input_shape, loss):
        
        inputs = Input(shape = (input_shape,))
        layer = Dense(32, activation = K.sigmoid)(inputs)
        lay = Dense(16,activation = K.sigmoid)(layer)
        out = Dense(1)(lay)              

        model = Model(inputs = inputs , outputs = out)

        model.compile(loss = loss, optimizer = 'adam')
       
        return model
		
		
#QMLP

def QMLP(x, input_shape, thau):
        
    def quantile_nonlinear(y_true,y_pred):
  
      x = y_true - y_pred
			#pretoze sa bude variac tensor, toto je postup pri kerase
  
      return K.maximum(thau * x,(thau - 1) * x)
    
    model = MLP_model(x, input_shape = input_shape, loss = quantile_nonlinear)
        
    return model