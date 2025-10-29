import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def logreg_learner(X, Y, sample_weight=None):
	if np.isclose(np.var(Y), 0):
		predictor = lambda x: np.ones((x.shape[0], 1)) * Y[0]
		is_const = True
	else:
		# Fit LogisticRegression in scaling pipeline
		pipe = Pipeline([('scale', StandardScaler()),
						 ('logreg', LogisticRegression(penalty='l2'))]).fit(X, Y, logreg__sample_weight=sample_weight)
		predictor = lambda x: pipe.predict_proba(x)[:,[1]]
		# Check if model is constant
		is_const = np.isclose(np.var(pipe["logreg"].coef_), 0)
	return predictor, {"is_const": is_const}

def logregl1_learner(X, Y, sample_weight=None):
	if np.isclose(np.var(Y), 0):
		predictor = lambda x: np.ones((x.shape[0], 1)) * Y[0]
		is_const = True
	else:
		# Fit LogisticRegression in scaling pipeline
		pipe = Pipeline([('scale', StandardScaler()),
						 ('logreg', LogisticRegression(penalty='l1', solver='saga'))]).fit(X, Y, logreg__sample_weight=sample_weight)
		predictor = lambda x: pipe.predict_proba(x)[:,[1]]
		# Check if model is constant
		is_const = np.isclose(np.var(pipe["logreg"].coef_), 0)
	return predictor, {"is_const": is_const}

def make_logregcv_learner(penalty='l1', cv=5, default_logreg=True, random_seed=None):
	'''
	Make LogisticRegressionCV learner with given parameters
	If default_logreg = True, then if predictor is constant we default to logreg_learner with default l2 penalty
	'''
	@ignore_warnings(category=ConvergenceWarning)
	def logregcv_learner(X, Y, sample_weight=None):
		if np.isclose(np.var(Y), 0):
			predictor = lambda x: np.ones((x.shape[0], 1)) * Y[0]
			is_const = True
		else:
			# Fit LogisticRegressionCV in scaling pipeline
			pipe = Pipeline([('scale', StandardScaler()),
							 ('logregcv', LogisticRegressionCV(cv=cv, penalty=penalty, solver='saga', random_state=random_seed))]).fit(X, Y, logregcv__sample_weight=sample_weight)
			# Check if model is constant
			is_const = np.isclose(np.var(pipe["logregcv"].coef_), 0)
			if is_const and default_logreg:
				# Overwrite with the output of logreg_learner
				predictor, is_const = logreg_learner(X, Y, sample_weight=sample_weight)
			else:
				# Make sure predictor inverts Y scale transform
				predictor = lambda x: pipe.predict_proba(x)[:,[1]]
		return predictor, {"is_const": is_const}
	return logregcv_learner

