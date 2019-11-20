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
from rbf import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
from Evaluation import mean_squared_error,trimmed_mean_squares

class  NeuralNetowrkTraining(QuantileNetwork):
	
	def __init__(self,train_x,train_y,test_x = None,test_y = None, thau = 0.85):
	
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.thau = thau
	
	def IQR_QRBF(self, units = 40,betas = 2.0, epochs = 50, verbose = 0):
		
		self.units = units
		self.betas = betas
		self.shape = train_x.shape[1]
		self.epochs = epochs
		self.verbose = verbose
		obj = QuantileNetwork(x = self.train_x,units = self.units, betas = self.betas,input_shape = train_x.shape[1],
                          thau = self.thau)
		upper_qrbf = obj.upper_rbf()
		lower_qrbf = obj.lower_rbf()
		
		upper_qrbf.fit(self.train_x,self.train_y,epochs = self.epochs, verbose = self.verbose)
		lower_qrbf.fit(self.train_x,self.train_y,epochs = self.epochs, verbose = self.verbose)
		
		predsU = upper_qrbf.predict(self.train_x)
		predsL = lower_qrbf.predict(self.train_x)
		
		indR = np.zeros(self.train_x.shape[0], dtype = np.int32)
		
		for i in range(self.train_x.shape[0]):
			indR[i] = np.where((self.train_y[i] <= predsU[i]) & (self.train_y[i] >= predsL[i]),np.int64(i),1000)
		
		indR = indR[indR != 1000]
		train_xR = self.train_x[indR,:]
		train_yR = self.train_y[indR]
		TRBF = obj.RBF_model(x = train_xR, units = self.units,betas = self.betas,
                         loss = 'mean_squared_error',input_shape = self.shape)
		TRBF.fit(train_xR,train_yR,epochs = self.epochs,verbose = self.verbose)
    
		predsTRBF = TRBF.predict(self.test_x)
		
		return predsTRBF
		
	def IQR_QMLP(self, epochs = 50,verbose = 0):
		
		self.epochs = epochs
		self.vebose = verbose
		obj = QuantileNetwork(x = self.train_x,units = self.units, betas = self.betas,input_shape = self.shape, thau = self.thau)
		upper_qmlp = self.upper_mlp()
		lower_qmlp = self.lower_mlp()
		print(obj)
		
		upper_qmlp.fit(self.train_x,self.train_y,epochs = self.epochs, verbose = self.verbose)
		lower_qmlp.fit(self.train_x,self.train_y,epochs = self.epochs, verbose = self.verbose)
		
		predsU = upper_qmlp.predict(self.train_x)
		predsL = lower_qmlp.predict(self.train_x)
		
		indM = np.zeros(self.train_x.shape[0], dtype = np.int32)
		
		for i in range(self.train_x.shape[0]):
			indM[i] = np.where((self.train_y[i] <= predsU[i]) & (self.train_y[i] >= predsL[i]),np.int64(i),1000)
		
		indM = indM[indM != 1000]
		train_xM = self.train_x[indM,:]
		train_yM = self.train_y[indM]
		
		TMLP = obj.MLP_model(loss = 'mean_squared_error',input_shape = self.shape)
		TMLP.fit(train_xM,train_yM,epochs = self.epochs,verbose = self.verbose)
    
		predsTMLP = TMLP.predict(self.test_x)
		
		return predsTMLP
		
		
	def evaluate(self,func,y_true,y_pred):
    
		if func == 'mean_squared_error':
      
			return mean_squared_error(y_true,y_pred)
    
		elif func == 'trimmed_mean_squared_error':
      
			return trimmed_mean_squares(y_true,y_pred,alpha = 0.75)
			
	def final_df(self,TMSE_RBF,TMSE_MLP,TMSE_TRBF,TMSE_TMLP,MSE_RBF,MSE_MLP,MSE_TRBF,MSE_TMLP):
	
		d = {'TMSE': [TMSE_RBF,TMSE_TRBF,TMSE_MLP,TMSE_TMLP], 'MSE': [MSE_RBF,MSE_TRBF,MSE_MLP,MSE_TMLP]}
		df = p.DataFrame(data = d)
		return df.rename(index ={0:'RBF',1:'TRBF',2: 'MLP', 3: 'TMLP'})
		
	def cross_val(self,K):
		
		KFold_split = KFold(n_splits = K, shuffle = True)
		
		#metrics
		
		cv_mse_rbf = np.zeros(K,dtype = np.int64)
		cv_mse_trbf = np.zeros(K,dtype = np.int64)
		cv_mse_mlp = np.zeros(K,dtype = np.int64)
		cv_mse_tmlp = np.zeros(K,dtype = np.int64)
		cv_tmse_rbf = np.zeros(K,dtype = np.int64)
		cv_tmse_trbf = np.zeros(K,dtype = np.int64)
		cv_tmse_mlp = np.zeros(K,dtype = np.int64)
		cv_tmse_tmlp = np.zeros(K,dtype = np.int64)
		
		m = 0
		
		for train,test in KF.split(train_x):
		
			#indices
			train_x_cv = self.train_x[train,:]
			train_y_cv = self.train_y[train]
			test_x_cv = self.train_x[test,:]
			test_y_cv = self.train_x[test]
			
			obj = NeuralNetowrkTraining(train_x = train_x_cv, train_y = train_y_cv, test_x =  test_x_cv, test_y =  test_y_cv)
			predsTRBF = obj.IQR_QRBF()
			predsTMLP = obj.IQR_QMLP()
			
			obj2 = QuantileNetwork(x = train_x_cv,units = 40, betas = 2.0,input_shape = train_x_cv.shape[1],thau = 0.85)
			
			RBF = obj2.RBF_model(x = train_x_cv,units = 40, betas = 2.0,input_shape = train_x_cv.shape[1], loss = 'mean_squared_error')
			MLP = obj2.MLP_model(input_shape = train_x_cv.shape[1], loss = 'mean_squared_error')
      
			RBF.fit(train_x_cv,train_y_cv,epochs = 50,verbose = 0)
			MLP.fit(train_x_cv,train_y_cv,epochs = 50,verbose = 0)
			
			predsMLP = MLP.predict(test_x_cv)
			predsRBF = RBF.predict(test_x_cv)
			
			cv_mse_rbf[m] = self.evaluate(func = 'mean_squared_error', y_true = test_y_cv, y_pred = predsRBF)
			cv_mse_trbf[m] = self.evaluate(func = 'mean_squared_error', y_true = test_y_cv, y_pred = predsTRBF)
			cv_mse_mlp[m] = self.evaluate(func = 'mean_squared_error', y_true = test_y_cv, y_pred = predsMLP)
			cv_mse_tmlp[m] = self.evaluate(func = 'mean_squared_error', y_true = test_y_cv, y_pred = predsTMLP)
			cv_tmse_rbf[m] = self.evaluate(func = 'trimmed_mean_squared_error', y_true = test_y_cv, y_pred = predsRBF)
			cv_tmse_trbf[m] = self.evaluate(func = 'trimmed_mean_squared_error', y_true = test_y_cv, y_pred = predsTRBF)
			cv_tmse_mlp[m] = self.evaluate(func = 'trimmed_mean_squared_error', y_true = test_y_cv, y_pred = predsMLP)
			cv_tmse_tmlp[m] = self.evaluate(func = 'trimmed_mean_squared_error', y_true = test_y_cv, y_pred = predsTMLP)
			
			m = m + 1
			print('Cross validation iteration number: ', m + 1)
			
		TMSE_RBF = np.mean(cv_mse_rbf)
		TMSE_TRBF = np.mean(cv_mse_trbf)
		TMSE_MLP = np.mean(cv_mse_mlp)
		TMSE_TMLP = np.mean(cv_mse_tmlp)
		TMSE_RBF = np.mean(cv_tmse_rbf)
		TMSE_TRBF = np.mean(cv_tmse_trbf)
		TMSE_MLP = np.mean(cv_tmse_mlp)
		TMSE_TMLP = np.mean(cv_tmse_tmlp)
		
		print('CV result')
		return self.final_df(TMSE_RBF,TMSE_TRBF,TMSE_MLP,TMSE_TMLP,MSE_RBF,MSE_TRBF,MSE_MLP,MSE_TMLP)