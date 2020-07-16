""" estimators.wrapper

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

import pickle
import pandas as pd
import numpy as np

# for evaluating
from sklearn.metrics import r2_score, mean_squared_error

# for GuWrapper
import multiprocessing
import gc
import itertools

# for random sampling
import random

from .utils import *


"""
	Start of wrappers
"""

class PooledWrapper(object):
	""" Wrapper implementing Gu et al. (2020). After each month, training set increases by one.

	Args:
		month (int, optional): Optionally only train the model annually.
		n (int): Number of models to average over.
		train_size (int): How many months to use for training. Default: 213 which is number
			of months between Apr 1958 to Dec 1974.
		test_size (int): How many months to use for testing. Default: 144 which is number of
			months between Jan 1975 to Dec 1986.
		hyperparams (dict): Hyperparameters to test.
		n_cpu (int): Number of threads.
		n_samples (int, optional): Randomly sample from permutations.
			If none, search through all permutations.

	"""

	def __init__(self, month=None, train_size=213, test_size=144,
		hyperparams=None, n_cpu=4, n_samples=None):
		super(PooledWrapper, self).__init__()

		self.month = month
		self.train_size = train_size
		self.test_size = test_size
		self.hyperparams = hyperparams
		self.n_cpu = n_cpu
		self.n_samples = n_samples

		self.estimator = None

	def hyperparam_validate(self, train_x, train_y, test_x, test_y, estimator_fn, hyperparams):
		""" Search for best hyperparams.

		Args:
			train_x (pandas.DataFrame): Training set independent variables.
			train_y (pandas.DataFrame): Training set dependent variables.
			test_x (pandas.DataFrame): Testing set independent variables.
			test_y (pandas.DataFrame): Testing set dependent variables.
			estimator_fn (function): Estimator function.
			hyperparams (dict): Dictionary with hyperparameters.

		Returns:
			dict: Best hyperparameters.

		"""

		val_loss = np.inf
		results = []

		keys = []
		permutations = []

		for k in hyperparams:
			keys.append(k)
			v = hyperparams[k]

			if not isinstance(v, collections.Iterable):
				permutations.append([v])
			else:
				permutations.append(v)

		permutations = list(itertools.product(*permutations))
		n = len(permutations)

		for i in range(n):
			params = dict(zip(keys, permutations[i]))
			print('Testing iteration {} params {}'.format(i, params))

			estimator = estimator_fn(params)
			mse, estimator = estimator.fit_set(train_x, train_y, test_x, test_y)

			results.append((mse, params, estimator))

		# go through the list and find the best
		best_mse = np.inf
		best_params = None
		best_estimator = None

		for mse, params, estimator in results:
			if mse < best_mse:
				best_mse = mse
				best_params = params
				best_estimator = estimator

		print('Best params {} loss {}', best_params, best_mse)

		return best_params, best_estimator

	def hyperparam_validate_multithreaded(self, train_x, train_y, test_x, test_y, estimator_fn,
		hyperparams):
		""" Search for best hyperparams.

		Args:
			train_x (pandas.DataFrame): Training set independent variables.
			train_y (pandas.DataFrame): Training set dependent variables.
			test_x (pandas.DataFrame): Testing set independent variables.
			test_y (pandas.DataFrame): Testing set dependent variables.
			estimator_fn (function): Estimator function.
			hyperparams (dict): Dictionary with hyperparameters.

		Returns:
			dict: Best hyperparameters.

		"""

		val_loss = np.inf
		results = []

		keys = []
		permutations = []

		for k in hyperparams:
			keys.append(k)
			v = hyperparams[k]

			if not isinstance(v, collections.Iterable):
				permutations.append([v])
			else:
				permutations.append(v)

		permutations = list(itertools.product(*permutations))

		if self.n_samples:
			permutations = random.sample(permutations, k=self.n_samples)
			print("Selected {}".format(permutations))

		n = len(permutations)

		with multiprocessing.Pool(processes=min(n, self.n_cpu)) as pool:
			for i in range(n):
				params = dict(zip(keys, permutations[i]))
				print('Testing iteration {} params {}'.format(i, params))

				estimator = estimator_fn(params)
				mse = pool.apply_async(estimator.fit_set, (train_x, train_y, test_x, test_y))

				results.append((mse, params))

			# go through the list and find the best
			best_mse = np.inf
			best_params = None
			best_estimator = None

			for mse, params in results:
				mse, estimator = mse.get()
				if mse < best_mse:
					best_mse = mse
					best_params = params
					best_estimator = estimator

		print('Best params {} loss {}'.format(best_params, best_mse))

		return best_params, best_estimator


	def fit_predict(self, df, estimator_fn, x_column, y_column, fill=None, forward=1,
		quote_date=None):
		""" Wrapper around a dataframe with first index a quote_date and evaluate the estimator
		for correlation, mse and r2.

		Args:
			df (pandas.DataFrame): DataFrame with index on quote_date.
			estimator_fn (estimator): Function returning an estimator.
			x_column (str or list(str)): Columns to use as X.
			y_column (str or list(str)): Column to use as y.
			fill (float): Fill nan.
			forward (int): Offset the training <-> prediction set by `forward`.
			quote_date (datetime): Date of this iteration. Used to determine whether to do
				hyperparameter search.

		Returns:
			pandas.Series: Result of estimator.predict(X).

		"""

		if isinstance(x_column, str):
			x_column = [x_column]

		if isinstance(y_column, str):
			y_column = [y_column]

		if fill is not None:
			df = df.fillna(0.)

		ts = df.index.get_level_values(level=0).unique().sort_values()

		# are there enough data?
		if len(ts) < 2:
			return None

		# split it into train and validation set
		if len(ts) < self.test_size + self.train_size:
			print('Data length {} is less than train size {} and test size {}'.format(len(ts),
				self.train_size, self.test_size))
			return None

		# indices for start/end of training/validation sets and prediction set
		train_start = ts[0]
		train_end = ts[self.train_size-1]
		test_start = ts[self.train_size]
		test_end = ts[min(self.train_size + self.test_size - 1, len(ts)-forward)]
		predict_idx = ts[-1]
		print('Train {} to {}; test {} to {}; predict {}'.format(train_start, train_end, test_start, test_end, predict_idx))

		train_set = df.loc[train_start:train_end]
		test_set = df.loc[test_start:test_end]
		predict_set = df.loc[[predict_idx]]
		print('len(train) {}; len(test) {}; len(predict) {}'.format(train_set.shape[0], test_set.shape[0], predict_set.shape[0]))

		self.train_size += 1

		train_x = train_set[x_column]
		train_y = train_set[y_column]
		test_x = test_set[x_column]
		test_y = test_set[y_column]

		if fill is not None:
			train_x = train_x.fillna(fill)
			train_y = train_y.fillna(fill)
			test_x = test_x.fillna(fill)
			test_y = test_y.fillna(fill)

		# try and save some memory
		df = None
		train_set = None
		test_set = None
		gc.collect()

		try:
			if (self.estimator is None
				or self.month is None
				# if we are training on real data, quote_date is a datetime
				or (self.month is not None
					and (isinstance(quote_date, np.datetime64) or isinstance(quote_date, pd.Timestamp))
					and self.month == quote_date.month)
				# if we are training on simulation, quote_date is an int
				or (self.month is not None
					and isinstance(quote_date, np.int64)
					and quote_date % 10 == self.month)):

				if self.n_cpu > 1:
					params, self.estimator = self.hyperparam_validate_multithreaded(train_x, train_y,
						test_x, test_y, estimator_fn, self.hyperparams)
				else:
					params, self.estimator = self.hyperparam_validate(train_x, train_y,
						test_x, test_y, estimator_fn, self.hyperparams)

			else:
				print('Skipping training')

			X = predict_set[x_column]

			predicted = self.estimator.predict(X)
			predicted = pd.Series(predicted, index=predict_set.index)

			return predicted

		except:
			print('Dumping to train_set.pkl and test_set.pkl')
			pickle.dump(train_set, open('train_set.pkl', 'wb'))
			pickle.dump(test_set, open('test_set.pkl', 'wb'))
			raise


def train_validate(ts, xs, ys, estimator, batch_size=2, thread=None):
	""" Loops through the data for one estimator, training and validating.

	Args:
		ts (list(datetime)): List of quote dates corresponding to index of xs and ys.
		xs (list(numpy.array)): List of numpy arrays in order.
		ys (list(numpy.array)): List of numpy arrays in order.
		estimator (function): Estimator.
		batch_size (int): Size of each training batch.

	Returns:
		tuple(pandas.DataFrame, estimator): Estimator used and list of validation loss.

	"""

	val_loss = []

	try:

		for t in range(batch_size+1, len(ts)):
			print('HP search iteration {}'.format(t))

			x = xs[t-batch_size-1:t-1]
			y = ys[t-batch_size-1:t-1]

			estimator = estimator.fit_validate(x, y)

			x = xs[t]
			y = ys[t]

			y_ = estimator.predict(x)

			if y_ is not None:
				val = mean_squared_error(y, y_)
				val_loss.append([ts[t], val])

		val_loss = pd.DataFrame(val_loss, columns=['quote_date', 'val_loss'])

		return val_loss, estimator

	except:
		print('Saving to xs.pkl, ys.pkl')
		pickle.dump(xs, open('xs.pkl', 'wb'))
		pickle.dump(ys, open('ys.pkl', 'wb'))
		raise


class OnlineDataFrameWrapper(object):
	""" Generic wrapper for iterating through a dataframe. Based on the first level of index.

	Args:
		hyperparams (dict): Hyperparameters to try.
		n_cpu (int): Number of multithreads.

	"""

	def __init__(self, hyperparams=None, n_cpu=4):
		super(OnlineDataFrameWrapper, self).__init__()

		self.hyperparams = hyperparams
		self.n_cpu = n_cpu

	def hyperparam_validate(self, df, x_column, y_column, estimator_fn, validation_start,
		train_size=1, n_samples=None, hyperparams=None):
		""" Search for best hyperparams.

		Args:
			df (pandas.DataFrame): Data.
			x_column (str or list(str)): Independent variables.
			y_column (str): Dependent variable.
			estimator_fn (function): Estimator function.
			validation_start (datetime): Date of validation start to the end of data set.
			train_size (int): Size of each batch.
			n_samples (int, optional): Randomly sample n choices from permutation.
			hyperparams (dict): Dictionary with hyperparameters.

		Returns:
			dict: Best hyperparameters.

		"""

		if isinstance(x_column, str):
			x_column = [x_column]

		if isinstance(y_column, str):
			y_column = [y_column]

		ts = df.index.get_level_values(level=0).unique().sort_values()
		batch_size = train_size * 2

		# are there enough data?
		if len(ts) < batch_size:
			return None

		xs = []
		ys = []
		for _, data in df.groupby(level=0):
			xs.append(data[x_column].values)
			ys.append(data[y_column].values)

		results = []
		keys = []
		permutations = []

		for k in hyperparams:
			keys.append(k)
			v = hyperparams[k]

			if not isinstance(v, collections.Iterable):
				permutations.append([v])
			else:
				permutations.append(v)

		permutations = list(itertools.product(*permutations))

		if n_samples:
			permutations = random.sample(permutations, k=n_samples)
			print("Selected {}".format(permutations))

		n = len(permutations)
		val_losses = []

		# go through the list and find the best
		best_loss = np.inf
		best_params = None
		best_estimator = None

		for i in range(n):
			params = dict(zip(keys, permutations[i]))
			print('Testing iteration {} params {}'.format(i, params))

			estimator = estimator_fn(params)
			estimator.set_hp_mode(True)
			val_loss, estimator = train_validate(ts, xs, ys, estimator, batch_size, i)

			estimator.set_hp_mode(False)
			loss = val_loss.loc[val_loss['quote_date'] >= validation_start, 'val_loss'].mean()
			val_losses.append((loss, params))

			if loss < best_loss:
				best_loss = loss
				best_params = params
				best_estimator = estimator

		print(val_losses)
		print('Best params {} loss {}'.format(best_params, best_loss))

		return best_params, best_estimator

	def hyperparam_validate_multithreaded(self, df, x_column, y_column, estimator_fn,
		validation_start, train_size=1, n_samples=None, hyperparams=None):
		""" Search for best hyperparams.

		Args:
			df (pandas.DataFrame): Data.
			x_column (str or list(str)): Independent variables.
			y_column (str): Dependent variable.
			estimator_fn (function): Estimator function.
			validation_start (datetime): Date of validation start to the end of data set.
			train_size (int): Size of each batch.
			n_samples (int, optional): Randomly sample n choices from permutation.
			hyperparams (dict): Dictionary with hyperparameters.

		Returns:
			dict: Best hyperparameters.

		"""

		if isinstance(x_column, str):
			x_column = [x_column]

		if isinstance(y_column, str):
			y_column = [y_column]

		ts = df.index.get_level_values(level=0).unique().sort_values()
		batch_size = train_size * 2

		# are there enough data?
		if len(ts) < batch_size:
			return None

		xs = []
		ys = []
		for _, data in df.groupby(level=0):
			xs.append(data[x_column].values)
			ys.append(data[y_column].values)

		results = []
		keys = []
		permutations = []

		for k in hyperparams:
			keys.append(k)
			v = hyperparams[k]

			if not isinstance(v, collections.Iterable):
				permutations.append([v])
			else:
				permutations.append(v)

		permutations = list(itertools.product(*permutations))

		if n_samples:
			permutations = random.sample(permutations, k=n_samples)
			print("Selected {}".format(permutations))

		n = len(permutations)
		val_losses = []

		with multiprocessing.Pool(processes=min(n, self.n_cpu)) as pool:
			for i in range(n):
				params = dict(zip(keys, permutations[i]))
				print('Testing iteration {} params {}'.format(i, params))

				estimator = estimator_fn(params)
				estimator.set_hp_mode(True)
				result = pool.apply_async(train_validate, (ts, xs, ys, estimator,
					batch_size, i))

				results.append((result, params))

			# go through the list and find the best
			best_loss = np.inf
			best_params = None
			best_estimator = None

			for result, params in results:
				val_loss, estimator = result.get()
				estimator.set_hp_mode(False)
				loss = val_loss.loc[val_loss['quote_date'] >= validation_start, 'val_loss'].mean()
				val_losses.append((loss, params))

				if loss < best_loss:
					best_loss = loss
					best_params = params
					best_estimator = estimator

		print(val_losses)
		print('Best params {} loss {}'.format(best_params, best_loss))

		return best_params, best_estimator


	def fit_predict(self, df, estimator, x_column, y_column, fill=None, forward=1,
		quote_date=None, **kwargs):
		""" Generic wrapper around a dataframe with first index a quote_date.

		Args:
			df (pandas.DataFrame): DataFrame with index on quote_date.
			estimator (function): BaseEstimator compatible estimator.
			x_column (str or list(str)): Columns to use as X.
			y_column (str or list(str)): Column to use as y.
			fill (float): Fill nan.
			forward (int): Offset the training <-> prediction set by `forward`.
			quote_date (datetime): For record keeping.
			**kwargs: Keyword arguments for `estimator`.

		Returns:
			pandas.Series: Result of estimator.predict(X).

		"""

		if isinstance(x_column, str):
			x_column = [x_column]

		if isinstance(y_column, str):
			y_column = [y_column]

		ts = df.index.get_level_values(level=0).unique().sort_values()

		# are there enough data?
		if len(ts) < 2:
			return None

		if fill is not None:
			df = df.fillna(fill)

		train_set = df.loc[ts[0]:ts[-1-forward]]
		predict_set = df.loc[[ts[-1]]]

		X = train_set[x_column]
		y = train_set[y_column]

		try:
			estimator = estimator.fit(X, y)

			X = predict_set[x_column]
			predicted = estimator.predict(X)
			predicted = pd.DataFrame(predicted, index=predict_set.index, columns=y_column)

			return predicted

		except:
			if 'predicted' in locals():
				pickle.dump(predicted, open('predicted.pkl', 'wb'))

			if 'y' in locals():
				pickle.dump(y, open('y.pkl', 'wb'))

			if 'y_' in locals():
				pickle.dump(y_, open('y_.pkl', 'wb'))

			pickle.dump(X, open('x.pkl', 'wb'))
			pickle.dump(y, open('y.pkl', 'wb'))
			raise

