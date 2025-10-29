import numpy as np
import pandas as pd
import scipy.stats
import warnings
import time

from sklearn.model_selection import KFold



def dimension(X):
	return len(X.shape)

def lexicographic_gt(A, b):
	'''
	A: an n x k matrix
	b: a length k vector

	Return n-vector of booleans where ith element is true if A[i] > b lexicographically
	'''
	lex_gt = A[:,0] > b[0]
	lex_eq = A[:,0] == b[0]
	for j in range(1, A.shape[1]):
		lex_gt = lex_gt | (lex_eq & (A[:,j] > b[j]))
		lex_eq = lex_eq & (A[:,j] == b[j])
	return lex_gt

def causal_pseudo_outcome(T, X, Y, propensities, predY1=0, predY0=0):
	'''
	propensities, predY1, and predY0 should be float or vector
	predY1 and predY0 may also take the string "mean," in which case we use the intercept model

	Example: if we let propensities=0.5 and leave predY1 and predY0 as defaults, we get standard IPW in balanced RCT
	'''
	if isinstance(predY1, str) and predY1 == 'mean':
		predY1 = Y[T == 1].mean()
	if isinstance(predY0, str) and predY0 == 'mean':
		predY0 = Y[T == 0].mean()
	pY1 = predY1 + T * (Y - predY1) / propensities
	pY0 = predY0 + (1 - T) * (Y - predY0) / (1 - propensities)
	pY = pY1 - pY0
	return pY

def aipw_intercept_pseudo_outcome(T, X, Y, propensities="estimate", cv=5, random_seed=None):
	'''
	AIPW with cross-fitting and only using intercept model
	'''
	# Format
	if propensities == "estimate":
		propensities = T.mean()
	if not isinstance(propensities, np.ndarray):
		propensities = np.ones(len(Y)) * propensities
	# Initialize
	pY1 = np.zeros(len(Y))
	pY0 = np.zeros(len(Y))
	# Split
	kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
	for i, (train_index, test_index) in enumerate(kf.split(X)):
		# Estimate means on training samples
		m1, m0 = Y[train_index][T[train_index] == 1].mean(), Y[train_index][T[train_index] == 0].mean()
		# Calculate AIPW estimate on test samples
		pY1[test_index] = m1 + T[test_index] * (Y[test_index] - m1) / propensities[test_index]
		pY0[test_index] = m0 + (1 - T[test_index]) * (Y[test_index] - m0) / (1 - propensities[test_index])
	# Combine
	pY = pY1 - pY0
	return pY
