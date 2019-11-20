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
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
from TuneModel import TuneModel
from Datasets import datasets


if __name__ == "__main__":
  
     
      #eps = np.array([0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17, 
      #      0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30, 
      #      0.31,0.32,0.33,0.34,0.35])
      
      alpha = np.array([0.70,0.75,0.78,0.80,0.82,0.85])
      d = datasets()
      train_x,train_y = d.nonlinear_data(eps = None,author = 'PV1')
          
      for alpha in alpha:
            
        x_max_abs = np.max(np.abs(train_x), axis=0)
        train_x /= x_max_abs



        init = TuneModel(train_x = train_x,train_y = train_y,units = 50,
                        betas = 2.0)

        print(init.cross_val(K = 10, epochs = 150, model = 'all', file = 'table_alpha_' + str(alpha),alpha = alpha,
                            batch_size = 10))
