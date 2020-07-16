import os
import pickle

from estimators.wrapper import *
from estimators.neural_network import *
from estimators.utils import *


# Test 1: Use Online Early Stopping estimator
name = 'oes'
data = pickle.load(open('sim.pkl', 'rb'))

for n in range(10):
	fn = lambda params: OnlineEarlyStop(layers=(32, 16, 8), model_dir='tmp', num_epochs=100, batch_norm=True, batch_size=50,
		early_stop_patience=5, early_stop_tolerance=0.001, hidden_activation='relu', **params)
	d = OnlineDataFrameWrapper(n_cpu=12)
	# hyperparameter search
	df = data.loc[:120]
	x_column = exclude(data.columns, 'excess_return')
	y_column = 'excess_return'
	best_params, f = d.hyperparam_validate_multithreaded(df, x_column, y_column, fn, 60,
		hyperparams={
			'l1' : [10**-5, 10**-4, 10**-3],
			'learning_rate' : [0.001, 0.01]})
	# window = 2 + 1 because OES requires two time periods to train. The +1 is for prediction
	r = iterator_set(d.fit_predict, data, window=2+1, estimator=f,
		start_date=120,
		x_column=x_column,
		y_column=y_column)
	df = data.loc[120:]
	# r.index = df[y_column].index
	df['prediction'] = r
	pickle.dump(df, open('{}_{}.pkl'.format(name, n), 'wb'))

df.rename(columns={'prediction' : 'prediction_{}'.format(9)}, inplace=True)
for n in range(9):
	tmp = pickle.load(open('{}_{}.pkl'.format(name, n), 'rb'))
	df['prediction_{}'.format(n)] = tmp['prediction']
	os.remove('{}_{}.pkl'.format(name, n))

df['prediction'] = df[select_prefix(df.columns, 'prediction_')].mean(axis=1)
pickle.dump(df, open('{}.pkl'.format(name), 'wb'))


# Test 2: Use DNN estimator, replicating Gu et al. (2020)
# It is recommended to restart ipython in-between
name = 'dnn'
data = pickle.load(open('sim.pkl', 'rb'))

for n in range(10):
	fn = lambda params: KerasDNNEstimator(layers=(32, 16, 8), model_dir="tmp",
		batch_size=10000, num_epochs=100, batch_norm=True, early_stop_patience=5, early_stop_tolerance=0,
		hidden_activation='relu', **params)
	d = PooledWrapper(month=0, train_size=60, test_size=60, n_cpu=6,
		hyperparams={
			'l1' : [10**-5, 10**-4, 10**-3],
			'learning_rate' : [0.001, 0.01]})
	# hyperparameter search
	x_column = exclude(data.columns, 'excess_return')
	y_column = 'excess_return'
	# window=9999+1 because it's an expanding window. So just set it to a large number
	r = iterator_set(d.fit_predict, data, window=9999+1, min_window=60+60+1, estimator_fn=fn,
		start_date=120,
		x_column=x_column,
		y_column=y_column)
	df = data.loc[120:]
	# r.index = df[y_column].index
	df['prediction'] = r
	pickle.dump(df, open('{}_{}.pkl'.format(name, n), 'wb'))

df.rename(columns={'prediction' : 'prediction_{}'.format(9)}, inplace=True)
for n in range(9):
	tmp = pickle.load(open('{}_{}.pkl'.format(name, n), 'rb'))
	df['prediction_{}'.format(n)] = tmp['prediction']

df['prediction'] = df[select_prefix(df.columns, 'prediction_')].mean(axis=1)
pickle.dump(df, open('{}.pkl'.format(name), 'wb'))


# Test 3: Use DTS-SGD estimator, based on Aydore (2019)
name = 'dtssgd'
data = pickle.load(open('sim.pkl', 'rb'))

for n in range(10):
	fn = lambda params: DTSSGD(layers=(32, 16, 8), model_dir='tmp', batch_norm=True,
		window_size=10, a=0.95, hidden_activation='relu', **params)
	d = OnlineDataFrameWrapper(n_cpu=12)
	# hyperparameter search
	df = data.loc[:120]
	x_column = exclude(data.columns, 'excess_return')
	y_column = 'excess_return'
	best_params, f = d.hyperparam_validate_multithreaded(df, x_column, y_column, fn, 60,
		hyperparams={
			'l1' : [10**-5, 10**-4, 10**-3],
			'learning_rate' : [0.001, 0.01]})
	r = iterator_set(d.fit_predict, data, window=1+1, estimator=f,
		start_date=120,
		x_column=x_column,
		y_column=y_column)
	df = data.loc[120:]
	# r.index = df[y_column].index
	df['prediction'] = r
	pickle.dump(df, open('{}_{}.pkl'.format(name, n), 'wb'))

df.rename(columns={'prediction' : 'prediction_{}'.format(9)}, inplace=True)
for n in range(9):
	tmp = pickle.load(open('{}_{}.pkl'.format(name, n), 'rb'))
	df['prediction_{}'.format(n)] = tmp['prediction']
	os.remove('{}_{}.pkl'.format(name, n))

df['prediction'] = df[select_prefix(df.columns, 'prediction_')].mean(axis=1)
pickle.dump(df, open('{}.pkl'.format(name), 'wb'))

