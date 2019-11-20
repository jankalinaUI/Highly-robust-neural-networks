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
#from RBF_tf import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
#from NetworkTraining_keras import NeuralNetowrkTraining
#from Evaluation import mean_squared_error,trimmed_mean_squares
#from Losses import psi,quantile_nonlinear,least_weighted_square

class TuneModel:
	
	def __init__(self,train_x, train_y, test_x = None,test_y = None, 
					betas = 2.0,units = 40):
	
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.betas = betas
		self.units = units
		
	def cross_val(self,K,epochs, model, file,alpha):
		
		self.epochs = epochs
		KFold_split = KFold(n_splits = K, shuffle = True)
		
		#metrics
		
		if model == 'RBF':
		
			names = ['mse_rbf', 'mse_trbf','tmse_rbf', 'tmse_trbf']
		
		elif model == 'MLP':
			
			names = ['mse_mlp', 'mse_tmlp','tmse_mlp', 'tmse_tmlp']
		
		else:
			
			names = ['mse_rbf', 'mse_trbf', 'mse_mlp', 
					 'mse_tmlp', 'tmse_rbf', 'tmse_trbf', 
					 'tmse_mlp', 'tmse_tmlp']
		
		metrics = {}
		
		for i in range(len(names)):
			metrics[names[i]] = np.zeros(K,dtype = np.float64)
		
		m = 0
		
		print(str(K) + 'Fold Cross-Validation has started')
		print('Testing model: ' + model)
		
		for train,test in KFold_split.split(self.train_x):
		
			#indices to select train,test data
			train_x_cv = self.train_x[train,:]
			train_y_cv = self.train_y[train]
			test_x_cv = self.train_x[test,:]
			test_y_cv = self.train_y[test]
			
			#assign init values to object
			net_train = NeuralNetowrkTraining(train_x = train_x_cv, 
						train_y = train_y_cv, test_x = test_x_cv, 
						test_y = test_y_cv)
      
			#create IQR predicton
			if model == 'RBF':
				
				predsTRBF = net_train.IQR_QRBF(epochs = self.epochs, units = self.units)
				RBF = net_train.RBF_model(x = train_x_cv,units = self.units, 
									  betas = self.betas,
									  input_shape = train_x_cv.shape[1], 
									  loss = 'mean_squared_error')
				RBF.fit(train_x_cv,train_y_cv,epochs = self.epochs,verbose = 0)
				predsRBF = RBF.predict(test_x_cv)
				print(predsRBF)
				predsDICT = dict(predsRBF = predsRBF,predsTRBF = predsTRBF)
				keyPreds = ['predsRBF','predsTRBF']
			
				# store for every acrhitecture loss values
				for j in range(len(predsDICT)):
					idx = j + 2
					metrics[names[j]][m] = net_train.evaluate(func = 'mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]])
					metrics[names[idx]][m] = net_train.evaluate(func = 'trimmed_mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]])
			
			elif model == 'MLP':
				
				predsTMLP = net_train.IQR_QMLP(epochs = self.epochs)
				MLP = net_train.MLP_model(input_shape = train_x_cv.shape[1], 
									  loss = 'mean_squared_error')
				MLP.fit(train_x_cv,train_y_cv,epochs = self.epochs,verbose = 0)
				predsMLP = MLP.predict(test_x_cv)
				predsDICT = dict(predsMLP = predsMLP,predsTMLP = predsTMLP)
				keyPreds = ['predsMLP','predsTMLP']
				
				for j in range(len(predsDICT)):
					idx = j + 2
					metrics[names[j]][m] = net_train.evaluate(func = 'mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]])
					metrics[names[idx]][m] = net_train.evaluate(func = 'trimmed_mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]])
			
			else:
				
				predsTMLP = net_train.IQR_QMLP(epochs = self.epochs)
				predsTRBF = net_train.IQR_QRBF(epochs = self.epochs, units = self.units)
				#train RBF,MLP network
				RBF = net_train.RBF_model(x = train_x_cv,units = self.units, 
									  betas = self.betas,
									  input_shape = train_x_cv.shape[1], 
									  loss = 'mean_squared_error')
				MLP = net_train.MLP_model(input_shape = train_x_cv.shape[1], 
									  loss = 'mean_squared_error')
      
				RBF.fit(train_x_cv,train_y_cv,epochs = self.epochs,verbose = 0)
				MLP.fit(train_x_cv,train_y_cv,epochs = self.epochs,verbose = 0)
				predsMLP = MLP.predict(test_x_cv)
				predsRBF = RBF.predict(test_x_cv)
				predsDICT = dict(predsRBF = predsRBF,predsTRBF = predsTRBF, predsMLP = predsMLP,predsTMLP = predsTMLP)
				keyPreds = ['predsRBF','predsTRBF','predsMLP','predsTMLP']
				
				# store for every acrhitecture loss values
				for j in range(len(predsDICT)):
					idx = j + 4
					metrics[names[j]][m] = net_train.evaluate(func = 'mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]])
					metrics[names[idx]][m] = net_train.evaluate(func = 'trimmed_mean_squared_error',
										y_true = test_y_cv,y_pred = predsDICT[keyPreds[j]],alpha = alpha)
			
			
			
			
			
			
			m = m + 1
			print(str(K) + 'Fold Cross-Validation iteration number: ', m)
			
		#take a mean of every metric
		for i in range(len(metrics)):
			metrics[names[i]] = np.mean(metrics[names[i]])
		
		print(str(K) + 'Fold Cross-Validation has finished')

		# Create data frame of final results
		return net_train.final_df(metrics, model_evaluate = model, file = file)
	
	def grid_search(self,units = None,betas = None, epochs = 200, neurons1 = None, neurons2 = None):
	
		#self.units = units
		#self.betas = betas
		self.epochs = epochs
		self.neurons1 = neurons1
		self.neurons2 = neurons2
		print(neurons1)
		
		print('Grid search for neural network model applied. Parameter used: \nneurons1: %s \nneurons2 %s \nepochs %s' % (self.neurons1,self.neurons2,self.epochs))
		
		obj = NeuralNetowrkTraining(train_x = self.train_x, train_y = self.train_y, test_x = self.test_x, test_y = self.test_y)
		obj2 = QuantileNetwork(x = self.train_x, input_shape = self.train_x.shape[1], units = 40, betas = 2.0, thau = 0.85)
		
		#train_xR,train_yR = obj.IQR_QRBF(return_data = True, epochs = 200)
		train_xM,train_yM = obj.IQR_QMLP(return_data = True, epochs = 200)
		print(train_yM)
		
		#RBF = KerasRegressor(build_fn = obj2.RBF_model, epochs = 200)
		MLP = KerasRegressor(build_fn = obj2.MLP_model, epochs = 200)
		
		
		
		#paramsRBF = dict(units = units, betas = betas)
		paramsMLP = dict(neurons1 = self.neurons1, neurons2 = self.neurons2)
		
    #gridRBF = GridSearchCV(estimator=RBF, param_grid=paramsRBF)
		#grid_resultRBF = gridRBF.fit(train_xR,train_yR)
		#means = grid_resultRBF.cv_results_['mean_test_score']
		#stds = grid_resultRBF.cv_results_['std_test_score']
		#params = grid_resultRBF.cv_results_['params']
		#print("Best: %f using %s" % (grid_resultRBF.best_score_, grid_resultRBF.best_params_))
		#for mean, stdev, param in zip(means, stds, params):
		#	print("%f (%f) with: %r" % (mean, stdev, param))
		
		gridMLP = GridSearchCV(estimator=MLP, param_grid=paramsMLP)
		grid_resultMLP = gridMLP.fit(train_xM,train_yM)
		means = grid_resultMLP.cv_results_['mean_test_score']
		stds = grid_resultMLP.cv_results_['std_test_score']
		params = grid_resultMLP.cv_results_['params']
			
		print("Best: %f using %s" % (grid_resultMLP.best_score_, grid_resultMLP.best_params_))
		for mean, stdev, param in zip(means, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))
		
		
		