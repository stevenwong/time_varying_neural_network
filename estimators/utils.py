""" utils

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


# for exclude
import collections

import random
import string


"""
	Some utility functions
"""

def iterator_set(fn, data, window=None, min_window=None, start_date=None, **kwargs):
	""" Iterate through first level of dataframe and pass the entire content to `fn`.

	Args:
		fn (callable): Function to apply.
		x (pandas.DataFrame): Input data.
		window (int, optional): Optionally provide a window rather than just a slice.
		min_window (int, optional): Optionally provide a minimum window.
		start_date (datetime): Start date.

	Returns:
		np.array: Result.

	"""

	ts = data.index.get_level_values(level=0).unique().sort_values()
	results = []

	try:

		for i in range(len(ts)):
			t = ts[i]
			print('Iterating through {}'.format(t))

			if start_date and start_date > t:
				continue

			if window is None:
				df = data.loc[t]
			elif min_window and i+1 < min_window:
				continue
			else:
				# DataFrame.loc is inclusive
				start = ts[max(i-window+1, 0)]
				df = data.loc[start:t]

			result = fn(df, quote_date=t, **kwargs)

			if result is not None:
				results.append(result)

		results = pd.concat(results)

		return results

	except:
		print('Saving results to results.pkl')
		import pickle
		pickle.dump(results, open('results.pkl', 'wb'))
		raise


def exclude(from_list, to_exclude):
	if isinstance(from_list, str):
		from_list = [from_list]

	if not isinstance(to_exclude, collections.Iterable) or isinstance(to_exclude, str):
		to_exclude = [to_exclude]

	xs = []
	for x in from_list:
		if not x in to_exclude:
			xs.append(x)

	return xs

def random_str(n):
	return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

