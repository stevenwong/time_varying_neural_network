""" estimators.neural_network.py

Copyright (C) 2020 Steven Wong <steven.ykwong87@gmail.com>

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import pandas as pd
import numpy as np
import copy
import math

# using sklearn
from sklearn.base import BaseEstimator

# for early stopping
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import entropy

# turn off excessive debug statements
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# keras and tensorflow
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Conv1D, BatchNormalization, Dropout, GaussianNoise
from keras.layers import SimpleRNN,TimeDistributed, Lambda, Concatenate, AveragePooling1D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Nadam, SGD, RMSprop, Adagrad, Adam
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from keras import regularizers
from keras import losses
import keras.backend as K

# for saving models
import io
import h5py

from .utils import *


class KerasDNNEstimator(BaseEstimator):
	""" Building DNN with Keras.

	Args:
		layers (list(int)): Number of nodes in each hidden layer.
		output_activation (str, optional): Keras activation function for output layer.
		hidden_activation (str, optional): Activation for hidden layers. Same as output
			if none.
		batch_size (int): Minibatch size.
		num_epochs (int, optional): Number of epochs.
		learning_rate (float, optional): Initial learning rate. Default 0.01.
		batch_norm (boolean): Batch normalisation. Default False.
		l1 (float, optional): l1 penalty on weights.
		l2 (float, optional): l2 penalty on weights.
		model_dir (str, optional): Where to save models.
		early_stop_patience (int, optional): Number of epochs to wait until we call early
			stopping. Default None.
		early_stop_tolerance (float, optional): If loss doesn't change by more than tolerance
			then training is stopped. Default None.
		early_stop_split (str, optional): Split ratio for the testing set.
		input_dropout (float, optional): Drop out rate for input layer. Default None.
		hidden_dropout (float, optional): Drop out rate for hidden layers. Default None.
		loss (str, optional): 'mse' - mean squared error, 'logcosh' - log(cosh)
		debug (boolean, optional): Debug.

	"""

	def __init__(self, layers=(32, 16, 8), output_activation='linear', hidden_activation='relu`',
		batch_size=32, num_epochs=100, learning_rate=0.01, batch_norm=False, l1=None, l2=None,
		model_dir=None, optimizer='Adam', early_stop_patience=10, early_stop_tolerance=0.001,
		early_stop_split=0.25, input_dropout=None, hidden_dropout=None, loss='mse', debug=False):
		super(KerasDNNEstimator, self).__init__()

		self.layers = layers
		self.output_activation = output_activation
		self.hidden_activation = (hidden_activation if hidden_activation is not None
			else output_activation)
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.batch_norm = batch_norm
		self.l1 = l1
		self.l2 = l2
		self.model_dir = model_dir
		self.optimizer = optimizer
		self.early_stop_patience = early_stop_patience
		self.early_stop_tolerance = early_stop_tolerance
		self.early_stop_split = early_stop_split
		self.input_dropout = input_dropout
		self.hidden_dropout = hidden_dropout
		self.loss = loss
		self.debug = debug

		self.model = None
		self.weights = None
		self.model_file = None

	def input_train(self, X, y):
		""" Transforms the data into (number of batches, batch size, number of features).
		If data doesn't have enough data for any of the batch, it will be filled with zero.

		Args:
			X (pandas.DataFrame): Features in shape (no. batches * batch_size, features).
			y (pandas.Series): Labels.

		Returns:
			tuple(numpy.array, numpy.array): Input shaped into correct 3-D shape.

		"""

		return X.values, y.values

	def input_predict(self, X):
		""" Transforms the data into (1, a, b).

		Args:
			X (numpy.array): Features in shape (no. batches * batch_size, features).

		Returns:
			numpy.array: Input shaped into correct size.

		"""

		# return np.reshape(X.values, (-1, X.shape[0], X.shape[1]))
		return X.values

	def build_model(self, layers, input_dim=None, output_activation='linear',
		hidden_activation='tanh', batch_norm=False, input_dropout=None, hidden_dropout=None,
		learning_rate=0.01):
		""" Build the DNN specified in the parameters.

		Args:
			layers (tuple(int)): Dense layer configurations.
			input_dim (int): Input dimension.
			output_activation (str, optional): Keras activation function for output layers.
			hidden_activation (str, optional): Keras activation function for hidden layers.
			batch_norm (boolean): Batch normalisation.
			dropout (float, optional): Dropout rate.
			learning_rate (float, optional): Learning rate.

		"""

		input_layer = nn = Input(shape=(input_dim,))

		if input_dropout:
			nn = Dropout(input_dropout)(nn)

		for u in layers:
			if self.l1 and self.l2:
				reg = regularizers.l1_l2(l1=self.l1, l2=self.l2)
			elif self.l1:
				reg = regularizers.l1(self.l1)
			elif self.l2:
				reg = regularizers.l2(self.l2)
			else:
				reg = None

			nn = Dense(u,
				activation=hidden_activation,
				kernel_regularizer=reg)(nn)

			if hidden_dropout:
				nn = Dropout(hidden_dropout)(nn)

			if batch_norm:
				nn = BatchNormalization()(nn)

		output_layer = Dense(1, activation=output_activation)(nn)

		if self.optimizer == 'Nadam':
			opt = Nadam(lr=learning_rate)
		elif self.optimizer == 'SGD':
			opt = SGD(lr=learning_rate)
		elif self.optimizer == 'RMSprop':
			opt = RMSprop(lr=learning_rate)
		elif self.optimizer == 'Adagrad':
			opt = Adagrad(lr=learning_rate)
		elif self.optimizer == 'Adam':
			opt = Adam(lr=learning_rate)
		else:
			ValueError('Invalid optimizer ' + self.optimizer)

		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(optimizer=opt, loss=self.loss)

		return model

	def fit(self, x, y):
		""" Fit DNN.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		tf.reset_default_graph()
		K.clear_session()

		if self.model_dir and not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		# calculate split
		ts = x.index.get_level_values(0)
		n = len(ts)
		test_split = n - int(max(np.round(n * self.early_stop_split, decimals=0), 1))
		test_x = x.loc[ts[test_split]:]
		test_y = y.loc[ts[test_split]:]
		train_x = x.loc[:ts[test_split]]
		train_y = y.loc[:ts[test_split]]

		train_x, train_y = self.input_train(train_x, train_y)
		test_x, test_y = self.input_train(test_x, test_y)
		c = EarlyStopping(min_delta=self.early_stop_tolerance, patience=self.early_stop_patience,
			restore_best_weights=True)

		# if we are using early stopping, we need at least two data sets
		if self.early_stop_patience and len(train_x) < 2:
			return self

		self.model = model = self.build_model(layers=self.layers, input_dim=train_x.shape[1],
			output_activation=self.output_activation, hidden_activation=self.hidden_activation,
			batch_norm=self.batch_norm, input_dropout=self.input_dropout,
			hidden_dropout=self.hidden_dropout, learning_rate=self.learning_rate)

		# train the model
		self.result = model.fit(train_x, train_y, batch_size=self.batch_size, validation_data=(test_x, test_y),
			callbacks=[c], epochs=self.num_epochs)

		self.weights = model.get_weights()

		return self

	def fit_set(self, train_x, train_y, test_x, test_y):
		""" Fit DNN.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		tf.reset_default_graph()
		K.clear_session()

		if self.model_dir and not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		train_x, train_y = self.input_train(train_x, train_y)
		test_x, test_y = self.input_train(test_x, test_y)
		c = EarlyStopping(min_delta=self.early_stop_tolerance, patience=self.early_stop_patience,
			restore_best_weights=True)

		# if we are using early stopping, we need at least two data sets
		if self.early_stop_patience and len(train_x) < 2:
			return self

		self.model = model = self.build_model(layers=self.layers, input_dim=train_x.shape[1],
			output_activation=self.output_activation, hidden_activation=self.hidden_activation,
			batch_norm=self.batch_norm, input_dropout=self.input_dropout,
			hidden_dropout=self.hidden_dropout, learning_rate=self.learning_rate)

		# train the model
		result = model.fit(train_x, train_y, batch_size=self.batch_size, validation_data=(test_x, test_y),
			callbacks=[c], epochs=self.num_epochs)

		self.weights = model.get_weights()

		# hyperparameter search is done in parallel, so save the model and redraw it
		self.model_file = 'tmp/' + random_str(12) + '.h5'
		self.model.save(self.model_file)
		self.model = None

		tf.reset_default_graph()
		K.clear_session()		

		return min(result.history['val_loss']), self

	def predict(self, x):
		""" Predict using fitted model.

		Args:
			x (numpy.array): Features.

		Returns:
			numpy.array: Predicted y.

		"""

		if self.model is None:
			if self.model_file is not None:
				# restore weights
				self.model = load_model(self.model_file)
				self.model_file = None

		y_ = self.model.predict(self.input_predict(x), verbose=self.debug)
		# return np.reshape(y_, (x.shape[0], 1))
		return y_


class OnlineEarlyStop(BaseEstimator):
	""" Online Early Stopping.

	https://arxiv.org/abs/2003.02515

	Args:
		layers (list(int)): Number of nodes in each hidden layer.
		output_activation (str, optional): Keras activation function for output layer.
		hidden_activation (str, optional): Activation for hidden layers. Same as output
			if none.
		num_epochs (int, optional): Number of epochs.
		batch_size (int, optional): Batch size.
		learning_rate (float, optional): Initial learning rate. Default 0.01.
		batch_norm (boolean): Batch normalisation. Default False.
		model_dir (str, optional): Where to save models.
		early_stop_patience (int, optional): Number of epochs to wait until we call early
			stopping. Default None.
		early_stop_tolerance (float, optional): If loss doesn't change by more than tolerance
			then training is stopped. Default None.
		l1 (float, optional): L1 penalty.
		l2 (float, optional): L2 penalty.
		input_dropout (float, optional): Drop out rate if used. Default None.
		hidden_dropout (float, optional): Drop out rate if used. Default None.
		loss (str, optional): 'mse' - mean squared error, 'logcosh' - log(cosh)
		hp_mode (Boolean, optional): Hyperparameter search mode.
		debug (boolean, optional): Debug.

	"""

	def __init__(self, layers=(32, 16, 8), output_activation='linear', hidden_activation='tanh',
		num_epochs=100, batch_size=100, learning_rate=0.01, batch_norm=False, model_dir=None,
		early_stop_patience=None, early_stop_tolerance=0.01, optimizer='Adam',
		l1=None, l2=None, input_dropout=None, hidden_dropout=None, loss='mse',
		hp_mode=False, debug=False):
		super(OnlineEarlyStop, self).__init__()

		self.layers = layers
		self.output_activation = output_activation
		self.hidden_activation = (hidden_activation if hidden_activation is not None
			else output_activation)
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.batch_norm = batch_norm
		self.model_dir = model_dir
		self.early_stop_patience = early_stop_patience
		self.early_stop_tolerance = early_stop_tolerance
		self.optimizer = optimizer
		self.l1 = l1
		self.l2 = l2
		self.input_dropout = input_dropout
		self.hidden_dropout = hidden_dropout
		self.loss = loss
		self.hp_mode = hp_mode
		self.debug = debug

		self.n = 0
		self.estimated_epochs = self.num_epochs

		# restore weights before last training
		self.prev_weights = None
		self.weights = None

		self.model = None
		self.model_file = None
		self.norms = []

		self.epochs = []

	def input_train(self, X, y):
		""" Transforms the data into (number of batches, batch size, number of features).
		If data doesn't have enough data for any of the batch, it will be filled with zero.

		Args:
			X (pandas.DataFrame or list(numpy.array)): Features in shape (no. batches * batch_size, features).
				If list of arrays, that means it's already converted.
			y (pandas.Series or list(numpy.array)): Labels.
				If list of arrays, that means it's already converted.

		Returns:
			tuple(numpy.array, numpy.array): Input shaped into correct 3-D shape.

		"""

		if isinstance(X, list):
			return X, y

		grouped = X.groupby(level=0)
		# batch_size = int(grouped.apply(lambda x: x.shape[0]).max())

		new_X = []
		for name, group in grouped:
			v = group.values
			new_X.append(v)

		grouped = y.groupby(level=0)

		new_y = []
		for name, group in grouped:
			v = group.values
			new_y.append(v)

		return new_X, new_y

	def input_predict(self, X):
		""" Transforms the data into (1, a, b).

		Args:
			X (numpy.array): Features in shape (no. batches * batch_size, features).

		Returns:
			numpy.array: Input shaped into correct size.

		"""

		if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
			return X.values
		else:
			return X

	def get_optimiser(self, learning_rate):
		if self.optimizer == 'Nadam':
			return Nadam(lr=learning_rate)
		elif self.optimizer == 'SGD':
			return SGD(lr=learning_rate)
		elif self.optimizer == 'RMSprop':
			return RMSprop(lr=learning_rate)
		elif self.optimizer == 'Adagrad':
			return Adagrad(lr=learning_rate)
		elif self.optimizer == 'Adam':
			return Adam(lr=learning_rate)
		else:
			ValueError('Invalid optimizer ' + self.optimizer)

	def build_model(self, layers, input_dim=None, output_activation='linear',
		hidden_activation='tanh', batch_norm=False, input_dropout=None, hidden_dropout=None, learning_rate=0.01):
		""" Build the DNN specified in the parameters.

		Args:
			layers (tuple(int)): Dense layer configurations.
			input_dim (int): Input dimension.
			output_activation (str, optional): Keras activation function for output layers.
			hidden_activation (str, optional): Keras activation function for hidden layers.
			batch_norm (boolean): Batch normalisation.
			dropout (float, optional): Dropout rate.
			learning_rate (float, optional): Learning rate.

		"""

		input_layer = nn = Input(shape=(input_dim,))

		if input_dropout:
			nn = Dropout(input_dropout)(nn)

		for u in layers:

			if self.l1 and self.l2:
				reg = regularizers.l1_l2(l1=self.l1, l2=self.l2)
			elif self.l1:
				reg = regularizers.l1(self.l1)
			elif self.l2:
				reg = regularizers.l2(self.l2)
			else:
				reg = None

			nn = Dense(u, activation=hidden_activation,
				kernel_regularizer=reg)(nn)

			if hidden_dropout:
				nn = Dropout(hidden_dropout)(nn)

			if batch_norm:
				nn = BatchNormalization()(nn)

		output_layer = Dense(1, activation=output_activation)(nn)

		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(optimizer=self.get_optimiser(learning_rate), loss=self.loss)

		return model

	def sampler(self, batch_size, n):
		""" Returns indices to randomly sample from array given len of array and batch size.

		Args:
			batch_size (int): Size of each batch.
			n (int): Total data set size.

		Returns:
			list(list(int)): List of indices to index an array.

		"""

		x = np.array(range(n))
		np.random.shuffle(x)

		# only take the k samples that result in a full sample set
		k = int(n / batch_size)
		x_ = x[:k*batch_size]

		return np.split(x_, k)

	def train_model(self, model, train_x, train_y, num_epochs, early_stop_patience=None,
		early_stop_tolerance=None):
		""" Helper function to train the model. Optionally specify whether to stop early.
		Early stopping algorithm broadly follows Chapter 7 of Goodfellow (2006).

		Args:
			model (keras.models.Model): Model.
			train_x (list(numpy.array)): Training features.
			train_y (list(numpy.array)): Training response.
			num_epochs (int): Maximum number of epochs to train.
			early_stop_patience (int, optional): Whether to stop early. If set, use early stopping.
			early_stop_tolerance (float, optional): Change in loss required before calling for
				early stop.

		Returns:
			tuple or None: None if early stopping not use. Otherwise, (epoch, train loss, test loss).

		"""

		# for early stopping
		num_periods = len(train_x)
		train_epochs = num_epochs

		# reverse the previous training to test for best epoch
		if self.prev_weights is not None:
			model.set_weights(self.prev_weights)

		for i in range(num_periods):
			best_loss = np.inf
			eval_loss = np.inf
			prev_loss = np.inf
			k = 0
			best_epoch = 0
			best_weights = None
			indices = None

			model.compile(optimizer=self.get_optimiser(self.learning_rate), loss=self.loss)

			# evaluate once to set previous loss
			if i == 0:
				y_ = model.predict(train_x[i+1])
				eval_loss = mean_squared_error(train_y[i+1], y_)
				print("Initial loss - eval loss {:.4f}".format(eval_loss))

			for t in range(train_epochs):
				if not indices:
					indices = self.sampler(min(self.batch_size, train_x[i].shape[0]), train_x[i].shape[0])

				idx = indices.pop()

				train_loss = model.train_on_batch(train_x[i][idx,:], train_y[i][idx,:])

				# evaluate
				if i == 0:
					y_ = model.predict(train_x[i+1])
					eval_loss = mean_squared_error(train_y[i+1], y_)
					print("Period {} Epoch {}/{} - train loss {:.4f}, eval loss {:.4f}".format(
						i, t+1, num_epochs, train_loss, eval_loss))

					# early stopping
					if early_stop_patience:
						if eval_loss < best_loss:
							best_loss = eval_loss
							best_epoch = t+1
							best_weights = model.get_weights()

						if eval_loss - prev_loss > -early_stop_tolerance:
							k = k+1
							if k > early_stop_patience:
								print('Early stopping at epoch {}'.format(t))
								print('Best epoch {:.4f}, loss {:.4f}'.format(best_epoch, best_loss))
								model.set_weights(best_weights)
								self.prev_weights = best_weights
								self.epochs.append(best_epoch)

								# recursively update best epoch estimate
								self.estimated_epochs = ((self.estimated_epochs * self.n + best_epoch)
									/ (self.n + 1))
								# self.estimated_epochs = np.median(self.epochs)
								self.n += 1
								train_epochs = int(np.round(self.estimated_epochs))
								print('Estimated epochs {}'.format(self.estimated_epochs))

								# with open('epochs.txt', 'a') as f:
								# 	f.write('{}\n'.format(train_epochs))

								break

						else:
							k = 0

						prev_loss = eval_loss
				else:
					print('Period {} Epoch {}/{} - train loss {:.4f}'.format(i, t+1, num_epochs, train_loss))

	def reset(self):
		tf.reset_default_graph()
		K.clear_session()

		self.model = None

	def set_hp_mode(self, hp_mode):
		""" Tensorflow has a bug where networks from multiprocessing returned won't get recreated properly
		on the second attempt. See::

			`https://github.com/keras-team/keras/issues/13380`

		"""

		self.hp_mode = hp_mode

		if not hp_mode:
			self.model_file = None

	def _fit(self, x, y):
		""" Fit DNN.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		if self.model_dir and not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		self.reset()

		train_x, train_y = self.input_train(x, y)

		if isinstance(x, list):
			n_features = x[0].shape[-1]
		else:
			n_features = x.shape[-1]

		if self.model is None:
			self.model = self.build_model(layers=self.layers,
				input_dim=n_features,
				output_activation=self.output_activation,
				hidden_activation=self.hidden_activation,
				batch_norm=self.batch_norm,
				input_dropout=self.input_dropout,
				hidden_dropout=self.hidden_dropout,
				learning_rate=self.learning_rate)

		# train the model
		self.train_model(self.model, train_x, train_y, self.num_epochs,
			self.early_stop_patience, self.early_stop_tolerance)

		return self

	def fit(self, x, y):
		""" Fit DNN.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		return self._fit(x, y)

	def fit_validate(self, x, y):
		""" If this is in the hyperparameter search stage, keep a copy of weights.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		self._fit(x, y)

		if self.model is not None:
			self.weights = self.model.get_weights()

			self.model_file = os.path.join(self.model_dir, random_str(12) + '.h5')
			self.model.save(self.model_file)
			self.model = None

		return self

	def predict(self, x):
		""" Predict using fitted model.

		Args:
			x (numpy.array): Features.

		Returns:
			numpy.array: Predicted y.

		"""

		if self.model_file and self.hp_mode:
			print('Resetting model')
			self.reset()
			self.model = load_model(self.model_file)
			# self.model_file = None

		if self.model is None:
			print('Model not trained. Skipping')
			return None

		y_ = self.model.predict(self.input_predict(x), verbose=self.debug)

		# tensorflow has issues with returning a model in multiprocessing
		if self.hp_mode:
			self.model = None

		return y_


class DTSSGD(BaseEstimator):
	""" An online variant of neural network.

	`https://papers.nips.cc/paper/9011-dynamic-local-regret-for-non-convex-online-forecasting.pdf`

	Args:
		layers (list(int)): Number of nodes in each hidden layer.
		output_activation (str, optional): Keras activation function for output layer.
		hidden_activation (str, optional): Activation for hidden layers. Same as output
			if none.
		window_size (int, optional): Look back window.
		a (float): Exponentially weighted scale.
		learning_rate (float, optional): Initial learning rate. Default 0.01.
		batch_norm (boolean): Batch normalisation. Default False.
		model_dir (str, optional): Where to save models.
		l1 (float, optional): L1 penalty.
		l2 (float, optional): L2 penalty.
		input_dropout (float, optional): Drop out rate if used. Default None.
		hidden_dropout (float, optional): Drop out rate if used. Default None.
		loss (str, optional): 'mse' - mean squared error, 'logcosh' - log(cosh)
		hp_mode (Boolean, optional): Hyperparameter search mode.
		debug (boolean, optional): Debug.

	"""

	def __init__(self, layers=(32, 16, 8), output_activation='linear', hidden_activation='tanh',
		window_size=10, a=0.99, learning_rate=0.01, batch_norm=False,
		model_dir=None, optimizer='SGD',
		l1=None, l2=None, input_dropout=None, hidden_dropout=None, loss='mse',
		hp_mode=False, debug=False):
		super(DTSSGD, self).__init__()

		self.layers = layers
		self.output_activation = output_activation
		self.hidden_activation = (hidden_activation if hidden_activation is not None
			else output_activation)
		self.window_size = window_size
		self.a = a
		self.learning_rate = learning_rate
		self.batch_norm = batch_norm
		self.model_dir = model_dir
		self.optimizer = optimizer
		self.l1 = l1
		self.l2 = l2
		self.input_dropout = input_dropout
		self.hidden_dropout = hidden_dropout
		self.loss = loss
		self.hp_mode = hp_mode
		self.debug = debug

		# restore weights before last training
		self.prev_weights = None
		self.weights = None

		self.model = None
		self.model_file = None
		self.grad_list = []

	def input_train(self, X, y):
		""" Transforms the data into (number of batches, batch size, number of features).
		If data doesn't have enough data for any of the batch, it will be filled with zero.

		Args:
			X (pandas.DataFrame or list(numpy.array)): Features in shape (no. batches * batch_size, features).
				If list of arrays, that means it's already converted.
			y (pandas.Series or list(numpy.array)): Labels.
				If list of arrays, that means it's already converted.

		Returns:
			tuple(numpy.array, numpy.array): Input shaped into correct 3-D shape.

		"""

		if isinstance(X, list):
			return X[0], y[0]

		return X.values, y.values

	def input_predict(self, X):
		""" Transforms the data into (1, a, b).

		Args:
			X (numpy.array): Features in shape (no. batches * batch_size, features).

		Returns:
			numpy.array: Input shaped into correct size.

		"""

		if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
			return X.values
		elif isinstance(X, list):
			return X[0]
		else:
			return X

	def get_optimiser(self, learning_rate):
		return SGD(lr=learning_rate)

	def build_model(self, layers, input_dim=None, output_activation='linear',
		hidden_activation='tanh', batch_norm=False, input_dropout=None, hidden_dropout=None,
		learning_rate=0.01):
		""" Build the DNN specified in the parameters.

		Args:
			layers (tuple(int)): Dense layer configurations.
			input_dim (int): Input dimension.
			output_activation (str, optional): Keras activation function for output layers.
			hidden_activation (str, optional): Keras activation function for hidden layers.
			batch_norm (boolean): Batch normalisation.
			dropout (float, optional): Dropout rate.
			learning_rate (float, optional): Learning rate.

		"""

		input_layer = nn = Input(shape=(input_dim,))

		if input_dropout:
			nn = Dropout(input_dropout)(nn)

		for u in layers:

			if self.l1 and self.l2:
				reg = regularizers.l1_l2(l1=self.l1, l2=self.l2)
			elif self.l1:
				reg = regularizers.l1(self.l1)
			elif self.l2:
				reg = regularizers.l2(self.l2)
			else:
				reg = None

			nn = Dense(u, activation=hidden_activation,
				kernel_regularizer=reg)(nn)

			if hidden_dropout:
				nn = Dropout(hidden_dropout)(nn)

			if batch_norm:
				nn = BatchNormalization()(nn)

		output_layer = Dense(1, activation=output_activation)(nn)

		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(optimizer=self.get_optimiser(learning_rate), loss=self.loss)

		return model

	def sampler(self, batch_size, n):
		""" Returns indices to randomly sample from array given len of array and batch size.

		Args:
			batch_size (int): Size of each batch.
			n (int): Total data set size.

		Returns:
			list(list(int)): List of indices to index an array.

		"""

		x = np.array(range(n))
		np.random.shuffle(x)

		# only take the k samples that result in a full sample set
		k = int(n / batch_size)
		x_ = x[:k*batch_size]

		return np.split(x_, k)

	def train_model(self, model, X, y):
		""" Manually perform gradient descent. See this example:
		`https://stackoverflow.com/questions/51354186/how-to-update-weights-manually-with-keras`

		and

		`https://github.com/Timbasa/Dynamic_Local_Regret_for_Non-convex_Online_Forecasting_NeurIPS2019/blob/master/code/Optim/dtssgd.py`

		Args:
			model (keras.models.Model): Model.
			X (list(numpy.array)): Training features.
			y (list(numpy.array)): Training response.

		Returns:
			float: Training loss.

		"""

		# reverse the previous training to test for best epoch
		if self.prev_weights is not None:
			model.set_weights(self.prev_weights)
			model.compile(optimizer=self.get_optimiser(self.learning_rate), loss=self.loss)

		# sample minibatch
		# idx = self.sampler(self.batch_size, len(X[0]))
		# batch_X = X[idx,:]
		# batch_y = y[idx,:]

		# check that training is working
		pre_mse = mean_squared_error(y, model.predict(X))

		# calculate loss
		loss = losses.mean_squared_error(y, model.output)
		sess = K.get_session()

		# symbolic gradient
		gradients = K.gradients(loss, model.trainable_weights)

		# actual gradient
		evaluated_gradients = sess.run(gradients, feed_dict={model.input: X})

		if len(self.grad_list) == self.window_size:
			self.grad_list.pop(0)

		self.grad_list.append(copy.deepcopy(evaluated_gradients))

		for i in range(len(model.trainable_weights)):
			layer = model.trainable_weights[i]

			# work out what the weighted gradient should be
			sum_grad = 0
			denominator = 0

			for j in range(len(self.grad_list)):
				sum_grad += (math.pow(self.a, len(self.grad_list) - 1 - j) * self.grad_list[j][i])
				denominator += math.pow(self.a, len(self.grad_list) - 1 - j)

			# print(sum_grad)
			# print(denominator)

			result_grad = sum_grad / denominator
			# print(np.linalg.norm(self.learning_rate * sum_grad, 2))

			# actually update the gradient
			sess.run(tf.assign_sub(layer, self.learning_rate * result_grad))

		post_mse = mean_squared_error(y, model.predict(X))

		print("Pre {}/Post {}".format(pre_mse, post_mse))

		return pre_mse
		

	def reset(self):
		tf.reset_default_graph()
		K.clear_session()

		self.model = None

	def set_hp_mode(self, hp_mode):
		""" Tensorflow has a bug where networks from multiprocessing returned won't get recreated properly
		on the second attempt. See::

			`https://github.com/keras-team/keras/issues/13380`

		"""

		self.hp_mode = hp_mode

		if not hp_mode:
			self.model_file = None

	def fit(self, x, y):
		""" Fit DNN.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		if self.model_dir and not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		self.reset()

		train_x, train_y = self.input_train(x, y)

		if isinstance(x, list):
			n_features = x[0].shape[-1]
		else:
			n_features = x.shape[-1]

		if self.model is None:
			self.model = self.build_model(layers=self.layers,
				input_dim=n_features,
				output_activation=self.output_activation,
				hidden_activation=self.hidden_activation,
				batch_norm=self.batch_norm,
				input_dropout=self.input_dropout,
				hidden_dropout=self.hidden_dropout,
				learning_rate=self.learning_rate)

		# train the model
		self.train_model(self.model, train_x, train_y)

		return self

	def fit_validate(self, x, y):
		""" If this is in the hyperparameter search stage, keep a copy of weights.

		Args:
			x (numpy.array): Independent variables:
			y (numpy.array): Dependent variables.

		Returns:
			self

		"""

		self.fit(x, y)

		if self.model is not None:
			self.weights = self.model.get_weights()

			self.model_file = 'tmp/' + random_str(12) + '.h5'
			self.model.save(self.model_file)
			self.model = None

		return self

	def predict(self, x):
		""" Predict using fitted model.

		Args:
			x (numpy.array): Features.

		Returns:
			numpy.array: Predicted y.

		"""

		if self.model_file and self.hp_mode:
			print('Resetting model')
			self.reset()
			self.model = load_model(self.model_file)
			# self.model_file = None

		if self.model is None:
			print('Model not trained. Skipping')
			return None

		y_ = self.model.predict(self.input_predict(x), verbose=self.debug)

		# tensorflow has issues with returning a model in multiprocessing
		if self.hp_mode:
			self.model = None

		return y_

