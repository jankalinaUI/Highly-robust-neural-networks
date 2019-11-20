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
from keras.datasets import boston_housing
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as p
from Evaluation import mean_squared_error, trimmed_mean_squared_error
from Losses import psi,quantile_nonlinear,least_weighted_square
from QuantileNetwork_keras import QuantileNetwork

class NeuralNetowrkTraining(QuantileNetwork):

    def __init__(self, train_x, train_y, test_x=None, test_y=None, thau=0.85):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.thau = thau

    def IQR_QRBF(self, units=40, betas=2.0, epochs=50, verbose=0, 
						return_data=False, batch_size = 32):
        
        self.units = units
        self.betas = betas
        self.shape = self.train_x.shape[1]
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size

        obj = QuantileNetwork(x=self.train_x,
                              units=self.units,
                              betas=self.betas,
                              input_shape=self.shape,
                              thau=self.thau,
                              neurons1=None,
                              neurons2=None)

        upper_qrbf, lower_qrbf = obj.QRBF()

        # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        upper_qrbf.fit(self.train_x,
                       self.train_y,
                       epochs=self.epochs,
                       verbose=0,
                       batch_size=self.batch_size)
        # callbacks = [tensorboard_callback])

        lower_qrbf.fit(self.train_x,
                       self.train_y,
                       epochs=self.epochs,
                       verbose=0,
                       batch_size=self.batch_size)
        # callbacks = [tensorboard_callback])

        predsU = upper_qrbf.predict(self.train_x)
        predsL = lower_qrbf.predict(self.train_x)

        indR = np.zeros(self.train_x.shape[0], dtype=object)


        
        for i in range(self.train_x.shape[0]):
              indR[i] = np.where((self.train_y[i] <= predsU[i]) & 
              (self.train_y[i] >= predsL[i]), np.int64(i), np.inf)
                

        indR = indR[indR != np.inf].astype(np.int32)
        train_xR = self.train_x[indR, :]
        train_yR = self.train_y[indR]
        TRBF = obj.RBF_model(x=train_xR,
                     units=self.units,
                     betas=self.betas,
                     loss='mean_squared_error', input_shape=self.shape)

        TRBF.fit(train_xR, train_yR,
            epochs=self.epochs,
             verbose=0,
            batch_size=self.batch_size)
# callbacks = [tensorboard_callback],
# validation_data = (self.test_x,self.test_y))

        predsTRBF = TRBF.predict(self.test_x)

        if return_data == False:
            return predsTRBF
        elif return_data == True:
            return train_xR, train_yR


    def IQR_QMLP(self, epochs=50, verbose=0, return_data=False, neurons1=None, neurons2=None):
          
          self.epochs = epochs
          self.vebose = verbose
          self.shape = self.train_x.shape[1]
          self.neurons1 = neurons1
          self.neurons2 = neurons2

          obj = QuantileNetwork(x=self.train_x,
                          units=None,
                          betas=None,
                          input_shape=self.shape,
                          thau=self.thau,
                          neurons1=self.neurons1,
                          neurons2=self.neurons2)

          upper_qmlp, lower_qmlp = self.QMLP()

          upper_qmlp.fit(self.train_x,
                   self.train_y,
                   epochs=self.epochs,
                   verbose=0, batch_size = self.batch_size)

          lower_qmlp.fit(self.train_x,
                   self.train_y,
                   epochs=self.epochs,
                   verbose=0, batch_size = self.batch_size)

          predsU = upper_qmlp.predict(self.train_x)
          predsL = lower_qmlp.predict(self.train_x)

          indM = np.zeros(self.train_x.shape[0], dtype=object)

          for i in range(self.train_x.shape[0]):
              indM[i] = np.where((self.train_y[i] <= predsU[i]) & 
              (self.train_y[i] >= predsL[i]), np.int64(i), np.inf)

          indM = indM[indM != np.inf].astype(np.int32)
          train_xM = self.train_x[indM, :]
          train_yM = self.train_y[indM]

          TMLP = obj.MLP_model(loss='mean_squared_error',
                         input_shape=self.shape,
                         neurons1=self.neurons1,
                         neurons2=self.neurons2)
          TMLP.fit(train_xM,
             train_yM,
             epochs=self.epochs,
             verbose=0, batch_size = self.batch_size)

          predsTMLP = TMLP.predict(self.test_x)

          if return_data == False:
              return predsTMLP
          elif return_data == True:
              return train_xM, train_yM


    def RBF_train(self, units=None, batch_size=32, betas=None, loss=None, epochs=100):
          
          self.units = units
          self.betas = betas
          self.loss = loss
          self.batch = batch_size
          self.epochs = epochs

          obj = QuantileNetwork(x=self.train_x,
                          units=self.units,
                          betas=self.betas,
                          input_shape=self.shape,
                          thau=self.thau)

          model = RBF_model(x=self.train_x,
                      input_shape=self.train_x.shape[1],
                      betas=self.betas,
                      units=self.units,
                      loss=self.loss)

          model.fit(self.train_x,
              self.train_y,
              epochs=self.epochs,
              batch_size=self.batch,
              verbose=0)

          return model


    def MLP_train(self, batch_size=32, loss=None, epochs=100):

          self.loss = loss
          self.batch = batch_size

          obj = QuantileNetwork(x=self.train_x,
                          units=self.units,
                          betas=self.betas,
                          input_shape=self.shape,
                          thau=self.thau,
                          neurons1 = None,
                          neurons2 = None)

          model = obj.MLP_model(x=self.train_x,
                      input_shape=self.train_x.shape[1],
                      loss=self.loss)

          model.fit(self.train_x,
              self.train_y,
              ecpohs=self.epochs,
              batch_size=self.batch,
              verbose=0)

          return model


    def evaluate(self, func, y_true, y_pred):
          
          if func == 'mean_squared_error':

              return mean_squared_error(y_true, y_pred)

          elif func == 'trimmed_mean_squared_error':

              return trimmed_mean_squared_error(y_true, y_pred, alpha=0.75)


    def final_df(self, dict_result, model_evaluate):
          
          if model_evaluate == 'RBF':
			
            metrics_trim = ['tmse_rbf', 'tmse_trbf']
            metrics_mean = ['mse_rbf', 'mse_trbf']
            d = {'TMSE': [dict_result[x] for x in metricsT],
				 'MSE': [dict_result[l] for l in metrics_]}
            df = p.DataFrame(data=d)
            df = df.rename(index={0: 'RBF', 1: 'TRBF'})
			
          elif model_evaluate == 'MLP':
            
            metrics_trim = ['tmse_mlp', 'tmse_tmlp']
            metrics_mean = ['mse_mlp', 'mse_tmlp']
            d = {'TMSE': [dict_result[x] for x in metricsT],
				 'MSE': [dict_result[l] for l in metrics_]}
            df = p.DataFrame(data=d)
            df = df.rename(index={0: 'MLP', 1: 'TMLP'})
          
		  else:
		  
            metrics_trim = ['tmse_rbf', 'tmse_trbf','tmse_mlp','tmse_tmlp']
            metrics_mean = ['mse_rbf', 'mse_trbf','mse_mlp','mse_tmlp']
            d = {'TMSE': [dict_result[x] for x in metricsT],
				 'MSE': [dict_result[l] for l in metrics_]}
            df = p.DataFrame(data=d)
            df = df.rename(index={0: 'RBF', 1: 'TRBF', 2: 'MLP', 3: 'TMLP'})
          
          
		  return df
