###############################################
#  10-CV FOLD FOR SIMULATED NONLINEAR DATA    #
###############################################



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

#evaluation functions
def mean_squared_error(y_true,y_pred):
    #residuals
    res = y_true - y_pred
    return np.mean(np.square(res))
    

def trimmed_mean_squared_error(y_true,y_pred,alpha = 0.75):
    res = np.sort(np.square(y_true - y_pred))
    h = np.int64(np.floor(alpha * len(y_true)))
    res = res[:h]
    return np.mean(res)

def data():
  X = np.arange(0,10,0.03)
  Y = np.sin(4 * X) + np.random.normal(0,1/2,len(X))
  #X = np.append(X,np.random.choice(np.arange(0,10,0.0,20, replace = False))
  X = np.append(X,np.arange(0,10,0.4))
  Y = np.append(Y,np.repeat(6,50))
  Y = Y[np.argsort(X,axis = 0)]
  X = np.sort(X,axis = 0)

  train_xx = X
  train_yy = Y

  train_xx = train_xx.reshape((len(train_xx),1))
  train_yy = train_yy.reshape((len(train_yy),1))
  
  return train_xx,train_yy


  
if __name__ == "__main__":
  
      sess = tf.Session() 
     
      train_x,train_y = data()
      
      KF = KFold(n_splits = 10, shuffle = True)
      
      cv_mse_trbf = np.zeros(10, dtype = np.float64)
      cv_mse_rbf = np.zeros(10, dtype = np.float64)
      cv_mse_tmlp = np.zeros(10, dtype = np.float64)
      cv_mse_mlp = np.zeros(10, dtype = np.float64)
      cv_tmse_trbf = np.zeros(10, dtype = np.float64)
      cv_tmse_rbf = np.zeros(10, dtype = np.float64)
      cv_tmse_tmlp = np.zeros(10, dtype = np.float64)
      cv_tmse_mlp = np.zeros(10, dtype = np.float64)
      
      
      
      
      dict_indices = {}
      
      k = 0
      
      for train,test in KF.split(train_x):
        
       
        train_cv = train_x[train,:]
        train_y_cv = train_y[train]
        test_cv = train_x[test,:]
        test_y_cv = train_y[test]
        
        
        
        dict_indices['train' + str(k)] = train
        dict_indices['test' + str(k)] = test
        
        modelQMLP1 = QMLP(x = train_cv,thau = 0.1, input_shape = 1)
        modelQMLP2 = QMLP(x = train_cv, thau = 0.9, input_shape = 1)
        modelQRBF1 = QRBF(x = train_cv,thau = 0.1,units = 40,betas = 2.0,input_shape = 1)
        modelQRBF2 = QRBF(x = train_cv, thau = 0.9,units = 40,betas = 2.0, input_shape = 1)
        modelMLP = MLP_model(train_cv, input_shape = 1, loss = 'mean_squared_error')
        modelRBF = RBF_model(train_cv, input_shape = 1, loss = 'mean_squared_error', units = 40, betas = 2.0)

        modelMLP.fit(train_cv,train_y_cv, batch_size = 10, epochs = 50, verbose = 0)

        modelRBF.fit(train_cv,train_y_cv, batch_size = 10, epochs = 50, verbose = 0)

        modelQMLP1.fit(train_cv,train_y_cv, batch_size = 10, epochs = 50, verbose = 0)

        modelQMLP2.fit(train_cv,train_y_cv, batch_size = 10, epochs = 50, verbose = 0)

        modelQRBF1.fit(train_cv,train_y_cv, batch_size = 10, epochs = 50, verbose = 0)

        modelQRBF2.fit(train_cv,train_y_cv, batch_size = 10, epochs = 50, verbose = 0)

        modelMLP = MLP_model(train_cv, input_shape = 1, loss = 'mean_squared_error')
        modelRBF = RBF_model(train_cv, input_shape = 1, loss = 'mean_squared_error', units = 40, betas = 2.0)

        predsQMLP1 = modelQMLP1.predict(train_cv)
        predsQMLP2 = modelQMLP2.predict(train_cv)
        predsQRBF1 = modelQRBF1.predict(train_cv)
        predsQRBF2 = modelQRBF2.predict(train_cv)
        predsMLP = modelMLP.predict(test_cv)
        predsRBF = modelRBF.predict(test_cv)


        indM = np.zeros(train_cv.shape[0], dtype = np.int32)
        indR = np.zeros(train_cv.shape[0], dtype = np.int32)
      
        for i in range(train_cv.shape[0]):
          indM[i] = np.where((train_y_cv[i] <= predsQMLP2[i]) & (train_y_cv[i] >= predsQMLP1[i]),
                         np.int64(i),1000)
          indR[i] = np.where((train_y_cv[i] <= predsQRBF2[i]) & (train_y_cv[i] >= predsQRBF1[i]),
                         np.int64(i),1000)
          
      
        indM = indM[indM != 1000]
        indR = indR[indR != 1000]
        train_xM = train_cv[indM,:]
        train_yM = train_y_cv[indM]
        train_xR = train_cv[indR,:]
        train_yR = train_y_cv[indR]

        modelTMLP = MLP_model(train_xM, input_shape = train_xM.shape[1], loss = 'mean_squared_error')
        modelTRBF = RBF_model(train_xR, input_shape = train_xR.shape[1], loss = 'mean_squared_error', units = 40, betas = 2.0)

        modelTMLP.fit(train_xM,train_yM,batch_size = 10, epochs = 50, verbose = 0)
        modelTRBF.fit(train_xR,train_yR,batch_size = 10, epochs = 50, verbose = 0)
        predsTMLP = modelTMLP.predict(test_cv)
        predsTRBF = modelTRBF.predict(test_cv)
        
        cv_mse_trbf[k] = mean_squared_error(test_y_cv,predsTRBF)
        cv_mse_rbf[k] = mean_squared_error(test_y_cv,predsRBF)
        cv_mse_tmlp[k] = mean_squared_error(test_y_cv,predsTMLP)
        cv_mse_mlp[k] = mean_squared_error(test_y_cv,predsMLP)
        cv_tmse_trbf[k] = trimmed_mean_squared_error(test_y_cv,predsTRBF)
        cv_tmse_rbf[k] = trimmed_mean_squared_error(test_y_cv,predsRBF)
        cv_tmse_tmlp[k] = trimmed_mean_squared_error(test_y_cv,predsTMLP)
        cv_tmse_mlp[k] = trimmed_mean_squared_error(test_y_cv,predsMLP)
        
        print(k)
        k = k + 1
