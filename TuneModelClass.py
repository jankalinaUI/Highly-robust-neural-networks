#Class for tuning neural network models

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

class TuneModel:
	
	def __init__(self,train_x, train_y, test_x,test_y):
	
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		
	def cross_val(self,K,epochs):
		
		self.epochs = epochs
		KFold_split = KFold(n_splits = K, shuffle = True)
		
		#metrics
		names = ['mse_rbf', 'mse_trbf', 'mse_mlp', 
				'mse_tmlp', 'tmse_rbf', 'tmse_trbf', 'tmse_mlp', 'tmse_tmlp']
		
		metrics = {}
		
		for i in range(len(names)):
			metrics[names[i]] = np.zeros(K,dtype = np.float64)
		
		m = 0
		
		for train,test in KFold_split.split(self.train_x):
		
			#indices
			train_x_cv = self.train_x[train,:]
			train_y_cv = self.train_y[train]
			test_x_cv = self.train_x[test,:]
			test_y_cv = self.train_y[test]
			
			net_train = NeuralNetowrkTraining(train_x = train_x_cv, train_y = train_y_cv)
			predsTRBF = net_train.IQR_QRBF(epochs = self.epochs)
			predsTMLP = net_train.IQR_QMLP(epochs = self.epochs)
			
			
			RBF = net_train.RBF_model(x = train_x_cv,units = 40, betas = 2.0,input_shape = train_x_cv.shape[1], loss = 'mean_squared_error')
			MLP = net_train.MLP_model(input_shape = train_x_cv.shape[1], loss = 'mean_squared_error')
      
			RBF.fit(train_x_cv,train_y_cv,epochs = 50,verbose = 0)
			MLP.fit(train_x_cv,train_y_cv,epochs = 50,verbose = 0)
			
			predsMLP = MLP.predict(test_x_cv)
			predsRBF = RBF.predict(test_x_cv)
			
			predsDICT = dict(predsRBF = predsRBF,predsTRBF = predsTRBF, predsMLP = predsMLP,predsTMLP = predsTMLP)
			keyPreds = ['predsRBF','predsTRBF','predsMLP','predsTMLP']
			
			for j in range(len(predsDICT)):
				idx = j + 4
				metrics[names[j]][m] = net_train.evaluate(func = 'mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]])
				metrics[names[idx]][m] = net_train.evaluate(func = 'trimmed_mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[idx]])
			
			
			m = m + 1
			print('Cross validation iteration number: ', m + 1)
			
		for i in range(len(metrics)):
			metrics[names[i]] = np.mean(metrics[names[i]])
		
		print('CV result')
		return net_train.final_df(metrics)
	
	def grid_search(self,units,betas, epochs):
	
		self.units = units
		self.betas = betas
		self.epochs = epochs
		
		print('Grid search for neural network model applied. Parameter used: \nunits: %s \nbetas %s \nepochs %s' % (self.units,self.betas,self.epochs))
		
		obj = NeuralNetowrkTraining(train_x = self.train_x, train_y = self.train_y, test_x = self.test_x, test_y = self.test_y)
		obj2 = QuantileNetwork(x = train_x)
		
		train_xR,train_yR = obj.IQR_QRBF(return_data = True, epochs = 100)
		train_xM,train_yM = obj.IQR_QMLP(return_data = True, epochs = 100)
		
		RBF = KerasRegressor(build_fn = obj2.RBF_model)
		MLP = KerasRegressor(build_fn = obj2.MLP_model)
		
		
		
		paramsRBF = dict(units = units, betas = betas, nb_epoch = epochs)
		paramsMLP = dict(nb_epochs = epochs)
		
		gridRBF = GridSearchCV(estimator=RBF, param_grid=paramsRBF)
		grid_resultRBF = gridRBF.fit(train_xR,train_yR)
		gridMLP = GridSearchCV(estimator=MLP, param_grid=paramsMLP)
		grid_resultMLP = gridMLP.fit(train_xM,train_yM)
		print("Best: %f using %s" % (grid_resultRBF.best_score_, grid_resultRBF.best_params_))
		for params, mean_score, scores in grid_resultRBF.grid_scores_:
			print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
			
		print("Best: %f using %s" % (grid_resultMLP.best_score_, grid_resultMLP.best_params_))
		for params, mean_score, scores in grid_resultMLP.grid_scores_:
			print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
		
		
		
		