import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def linreg_learner(X, Y, sample_weight=None):
	'''
	Linearly regress Y on X
	'''
	linreg = LinearRegression().fit(X, Y, sample_weight=sample_weight)
	predictor = lambda x: linreg.predict(x).reshape(-1,1)
	is_const = np.isclose(np.var(linreg.coef_), 0)
	return predictor, {"is_const": is_const}

def linreg_l2_learner(X, Y, sample_weight=None):
	'''
	Linearly regress Y on X with ridge
	'''
	ridge = Ridge().fit(X, Y, sample_weight=sample_weight)
	predictor = lambda x: ridge.predict(x).reshape(-1,1)
	is_const = np.isclose(np.var(ridge.coef_), 0)
	return predictor, {"is_const": is_const}

def ridgecv_learner(X, Y, sample_weight=None):
	'''
	RidgeCV with leave-one-out cross-validation
	'''
	# Scale Y
	Y_scaler = StandardScaler().fit(Y.reshape(-1,1))
	Y_std = Y_scaler.transform(Y.reshape(-1,1)).flatten()
	# Fit RidgeCV in scaling pipeline
	alphas = np.logspace(-5,5,20)
	pipe = Pipeline([('scale', StandardScaler()), ('ridgecv', RidgeCV(alphas=alphas))]).fit(X, Y_std, ridgecv__sample_weight=sample_weight)
	# Make sure predictor inverts Y scale transform
	predictor = lambda x: Y_scaler.inverse_transform(pipe.predict(x).reshape(-1,1)).reshape(-1,1)
	# Check if model is constant
	is_const = np.isclose(np.var(pipe["ridgecv"].coef_), 0)
	return predictor, {"is_const": is_const}

def make_lassocv_learner(cv=5, random_seed=None):
	'''
	Make LassoCV learner with given parameters
	'''
	@ignore_warnings(category=ConvergenceWarning)
	def lassocv_learner(X, Y, sample_weight=None):
		# Scale Y
		Y_scaler = StandardScaler().fit(Y.reshape(-1,1))
		Y_std = Y_scaler.transform(Y.reshape(-1,1)).flatten()
		# Fit LassoCV in scaling pipeline
		pipe = Pipeline([('scale', StandardScaler()), ('lassocv', LassoCV(cv=cv, random_state=random_seed))]).fit(X, Y_std, lassocv__sample_weight=sample_weight)
		# Make sure predictor inverts Y scale transform
		predictor = lambda x: Y_scaler.inverse_transform(pipe.predict(x).reshape(-1,1)).reshape(-1,1)
		# Check if model is constant
		is_const = np.isclose(np.var(pipe["lassocv"].coef_), 0)
		return predictor, {"is_const": is_const}
	return lassocv_learner

def make_elasticnetcv_learner(cv=5, random_seed=None):
	'''
	Make ElasticNetCV learner with given parameters
	'''
	@ignore_warnings(category=ConvergenceWarning)
	def elasticnetcv_learner(X, Y, sample_weight=None):
		# Scale Y
		Y_scaler = StandardScaler().fit(Y.reshape(-1,1))
		Y_std = Y_scaler.transform(Y.reshape(-1,1)).flatten()
		# Fit ElasticNetCV in scaling pipeline
		pipe = Pipeline([('scale', StandardScaler()), ('elasticnetcv', ElasticNetCV(cv=cv, random_state=random_seed))]).fit(X, Y_std, elasticnetcv__sample_weight=sample_weight)
		# Make sure predictor inverts Y scale transform
		predictor = lambda x: Y_scaler.inverse_transform(pipe.predict(x).reshape(-1,1)).reshape(-1,1)
		# Check if model is constant
		is_const = np.isclose(np.var(pipe["elasticnetcv"].coef_), 0)
		return predictor, {"is_const": is_const}
	return elasticnetcv_learner

def make_random_forest_learner(random_seed=None):
	'''
	Make RandomForestRegressor learner with given parameters
	'''
	def random_forest_learner(X, Y, sample_weight=None):
		rf = RandomForestRegressor(random_state=random_seed).fit(X, Y, sample_weight=sample_weight)
		predictor = lambda x: rf.predict(x).reshape(-1,1)
		return predictor, {}
	return random_forest_learner

def make_regularized_random_forest_learner(random_seed=None):
	'''
	Make RandomForestRegressor learner with shallower max depth and larger min_samples_leaf requirement
	'''
	def regularized_random_forest_learner(X, Y, sample_weight=None):
		rf = RandomForestRegressor(max_depth=2, min_samples_leaf=10, max_features=None, random_state=random_seed).fit(X, Y, sample_weight=sample_weight)
		predictor = lambda x: rf.predict(x).reshape(-1,1)
		return predictor, {}
	return regularized_random_forest_learner

def make_mlp_regressor_learner(random_seed=None):
	'''
	Make MLPRegressor (neural network) learner
	'''
	@ignore_warnings(category=ConvergenceWarning)
	def mlp_regressor_learner(X, Y, sample_weight=None):
		pipe = Pipeline([('scale', StandardScaler()), ('mlp', MLPRegressor(random_state=random_seed))]).fit(X, Y)
		predictor = lambda x: pipe.predict(x).reshape(-1,1)
		return predictor, {}
	return mlp_regressor_learner

def calculate_weighted_correlations(Y, X, sample_weight=None):
	'''
	Returns (possibly weighted) correlations between Y and every column of X
	'''
	# Process weights
	if sample_weight is None:
		w = np.ones(len(Y), dtype=float)
	else:
		w = sample_weight
	w_norm = w / w.sum()
	# Calculate weighted means
	mean_y = np.dot(w_norm, Y)
	mean_x = np.dot(w_norm, X)
	# Center data
	Yc = Y - mean_y
	Xc = X - mean_x
	# Calculated weighted second moments
	cov_xy = np.dot(w_norm, Xc * Yc[:, None])
	var_y  = np.dot(w_norm, Yc ** 2)
	var_x  = np.dot(w_norm, Xc ** 2)
	# Calculate correlations
	denom = np.sqrt(var_y * var_x)
	with np.errstate(divide="ignore", invalid="ignore"):
		corr = cov_xy / denom
		corr = np.where(denom == 0, 0.0, corr) # protect against 0 variance
	# Return
	return corr

def make_lassocv_selector_learner(cv=5, min_features=10, random_seed=None):
	'''
	Run LassoCV. If we have fewer than min_features in the final predictor, choose smallest alpha so that we have at least min_features.
	Note that sample_weight is ignored here.
	'''
	@ignore_warnings(category=ConvergenceWarning)
	def lassocv_selector_learner(X, Y, sample_weight=None):
		# Scale Y
		Y_scaler = StandardScaler().fit(Y.reshape(-1,1))
		Y_std = Y_scaler.transform(Y.reshape(-1,1)).flatten()
		# Fit LassoCV in scaling pipeline
		pipe = Pipeline([('scale', StandardScaler()), ('lassocv', LassoCV(cv=cv, random_state=random_seed))]).fit(X, Y_std)
		# Get Lasso coefficients; also get the scaler for later
		lasso_cv: LassoCV = pipe[-1]
		scaler = pipe[0]
		n_nonzero = np.count_nonzero(lasso_cv.coef_)
		if n_nonzero >= min_features:
			predictor = lambda x: Y_scaler.inverse_transform(pipe.predict(x).reshape(-1,1)).reshape(-1,1)
		else:
			# To enforce at least min_features, recompute whole regularization path
			X_std = scaler.transform(X)
			alphas, coefs, _ = LassoCV.path(X_std, Y_std, alphas=lasso_cv.alphas_)
			# Choose smallest alpha that leads to at least min_features
			try:
				idx = np.where((coefs != 0).sum(axis=0) >= min_features)[0][-1]
				alpha_sel = alphas[idx]
			except:
				# If there is no such feature, just choose the largest alpha
				alpha_sel = alphas[0]
			# Refit lasso
			lasso_pipe = make_pipeline(pipe[0], Lasso(alpha=alpha_sel)).fit(X, Y_std)
			predictor = lambda x: Y_scaler.inverse_transform(lasso_pipe.predict(x).reshape(-1,1)).reshape(-1,1)
		return predictor, {}
	return lassocv_selector_learner

