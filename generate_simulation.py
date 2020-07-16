import numpy as np
import pandas as pd
import pickle

np.random.seed(seed=42)

# constants
N = 200
T = 180
C = 100
p = 0.95

X = np.random.normal(0., 1., size=(T*N, C))
e = np.random.normal(0., 1., size=(T*N))
F = np.zeros((T, C))
y = np.zeros(T*N)

F[0,:] = np.random.normal(0., 1., size=C)

for t in range(1, T):
	F[t,:] = F[t-1,:] * p + (1 - p) * np.random.normal(0., 1., size=C)

for t in range(T):
	y[t*N:(t+1)*N] = np.tanh((X[t*N:(t+1)*N,:] * F[t,:])).sum(axis=1)

y = y + e

idx = np.zeros((T*N, 2))
for t in range(T):
	idx[t*N:(t+1)*N,0] = t
	idx[t*N:(t+1)*N,1] = list(range(N))

X = pd.DataFrame(X, columns=['feature.{}'.format(s) for s in range(C)])
X['date'] = idx[:,0]
X['date'] = X['date'].astype('int')
X['id'] = idx[:,1]
X['id'] = X['id'].astype('int')
X = X.set_index(['date', 'id'])
X['excess_return'] = y

pickle.dump(X, open('sim.pkl', 'wb'))
