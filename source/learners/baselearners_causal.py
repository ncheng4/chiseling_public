import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import ConstantInputWarning

# ----------------------------------------- Helper functions ----------------------------------------- #

def unimodal_fit(x, y, num_candidates=4):
	"""
	Unimodal regression with simple interior-mode grid + explicit global monotone fits.

	If x has only two distinct values, skips unimodal candidates and only
	compares globally increasing vs. decreasing isotonic fits.

	Returns
	-------
	f : callable
		Predictor that accepts scalar or array-like z. Uses clipped isotonic behavior.
	best_mse : float
		Training mean squared error of the selected model.
	"""
	# --- prep & sort ---
	x = np.asarray(x, float); y = np.asarray(y, float)
	m = np.isfinite(x) & np.isfinite(y)
	x, y = x[m], y[m]
	n = x.size
	if n == 0:
		return (lambda z: np.array([])), float("nan")
	if n == 1:
		c = float(y[0])
		return (lambda z: np.full_like(np.asarray(z, float), c, dtype=float)), 0.0
	o = np.argsort(x); x, y = x[o], y[o]
	xu = np.unique(x)
	binary_x = (xu.size <= 2)

	# --- helper: interior candidate indices from equally spaced x-grid ---
	def _grid_candidates(x, k):
		if k <= 0: return []
		xmin, xmax = x[0], x[-1]
		grid = np.linspace(xmin, xmax, k + 2)[1:-1]  # drop endpoints
		idxs = []
		for g in grid:
			i = np.searchsorted(x, g)
			if i <= 0: j = 0
			elif i >= n: j = n - 1
			else: j = i - 1 if abs(g - x[i-1]) <= abs(x[i] - g) else i
			if 0 < j < n - 1:
				idxs.append(j)
		return sorted(set(idxs))

	interior_modes = [] if binary_x else _grid_candidates(x, int(num_candidates))

	# --- candidates: (mse, predictor) ---
	candidates = []

	# 1) Explicit monotone increasing
	inc_model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
	yhat_inc = inc_model.predict(x)
	candidates.append((np.mean((y - yhat_inc) ** 2),
					   lambda z, m=inc_model: m.predict(np.asarray(z, float))))

	# 2) Explicit monotone decreasing
	dec_model = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
	yhat_dec = dec_model.predict(x)
	candidates.append((np.mean((y - yhat_dec) ** 2),
					   lambda z, m=dec_model: m.predict(np.asarray(z, float))))

	# 3) Stitched unimodal fits at interior modes (skipped if binary_x)
	for k in interior_modes:
		xk = x[k]
		L = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[:k+1], y[:k+1])
		R = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x[k:],  y[k:])
		mode_val = max(L.predict([xk])[0], R.predict([xk])[0])

		Lx = L.predict(x); Rx = R.predict(x)
		yhat = np.where(x <= xk, np.minimum(Lx, mode_val), np.minimum(Rx, mode_val))
		mse = np.mean((y - yhat) ** 2)

		def make_pred(L, R, xk, mv):
			def f(z):
				z = np.asarray(z, float)
				left  = np.minimum(L.predict(z), mv)
				right = np.minimum(R.predict(z), mv)
				return np.where(z <= xk, left, right)
			return f

		candidates.append((mse, make_pred(L, R, xk, mode_val)))

	# --- select best ---
	best_mse, best_f = min(candidates, key=lambda t: t[0])
	return best_f, float(best_mse)

# ----------------------------------------- Main learners ----------------------------------------- #

def causal_linreg_learner(TX, Y, sample_weight=None):
	T, X = TX[:,0], TX[:,1:]
	if np.isclose(np.var(T), 0):
		predictor = lambda tx: np.zeros((x.shape[0], 1))
		is_const = True
	else:
		linreg1 = LinearRegression().fit(X[T==1], Y[T==1], sample_weight=sample_weight)
		linreg0 = LinearRegression().fit(X[T==0], Y[T==0], sample_weight=sample_weight)
		predictor = lambda tx: (linreg1.predict(tx[:,1:]) - linreg0.predict(tx[:,1:])).reshape(-1,1)
		is_const = np.isclose(np.var(linreg.coef_), 0)
	return predictor, {"is_const": is_const}

def make_causal_random_forest_learner(random_seed=None):
	def causal_random_forest_learner(TX, Y, sample_weight=None):
		T, X = TX[:,0], TX[:,1:]
		if np.isclose(np.var(T), 0):
			predictor = lambda tx: np.zeros((tx.shape[0], 1))
			is_const = True
			meta = {"is_const": True}
		else:
			if sample_weight is not None:
				sw1, sw0 = sample_weight[T==1], sample_weight[T==0]
			else:
				sw1, sw0 = None, None
			rf1 = RandomForestRegressor(random_state=random_seed).fit(X[T==1], Y[T==1], sample_weight=sw1)
			rf0 = RandomForestRegressor(random_state=random_seed).fit(X[T==0], Y[T==0], sample_weight=sw0)
			predictor = lambda tx: (rf1.predict(tx[:,1:]) - rf0.predict(tx[:,1:])).reshape(-1,1)
			meta = {}
		return predictor, meta
	return causal_random_forest_learner

def make_causal_random_forest_classifier_learner(random_seed=None):
	def causal_random_forest_classifier_learner(TX, Y, sample_weight=None):
		T, X = TX[:,0], TX[:,1:]
		if np.isclose(np.var(T), 0):
			predictor = lambda tx: np.zeros((tx.shape[0], 1))
			is_const = True
			meta = {"is_const": True}
		else:
			if sample_weight is not None:
				sw1, sw0 = sample_weight[T==1], sample_weight[T==0]
			else:
				sw1, sw0 = None, None
			if np.isclose(Y[T==1].var(), 0):
				predictor_1 = lambda tx: Y[T==1][0] * np.ones((len(tx), 1))
			else:
				rf1 = RandomForestClassifier(random_state=random_seed).fit(X[T==1], Y[T==1], sample_weight=sw1)
				predictor_1 = lambda tx: rf1.predict_proba(tx[:,1:])[:,[1]]
			if np.isclose(Y[T==0].var(), 0):
				predictor_0 = lambda tx: Y[T==0][0] * np.ones((len(tx), 1))
			else:
				rf0 = RandomForestClassifier(random_state=random_seed).fit(X[T==0], Y[T==0], sample_weight=sw0)
				predictor_0 = lambda tx: rf0.predict_proba(tx[:,1:])[:,[1]]
			predictor = lambda tx: predictor_1(tx) - predictor_0(tx)
			meta = {}
		return predictor, meta
	return causal_random_forest_classifier_learner

def make_interpretable_causal_random_forest_classifier_learner(smoother="isotonic", random_seed=None):
	base_learner = make_causal_random_forest_classifier_learner(random_seed)
	@ignore_warnings(category=UserWarning)
	@ignore_warnings(category=ConstantInputWarning)
	def interpretable_causal_random_forest_classifier_learner(TX, Y, sample_weight=None):
		base_predictor, _ = base_learner(TX, Y, sample_weight=sample_weight)
		base_preds = base_predictor(TX).flatten()
		per_feat_predictors = []
		for i in range(TX[:,1:].shape[1]):
			feature = TX[:,1+i]
			if smoother == "isotonic":
				per_feat_predictor = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(feature.reshape(-1,1), base_preds, sample_weight=sample_weight)
			elif smoother == "linear":
				per_feat_predictor = LinearRegression().fit(feature.reshape(-1,1), base_preds, sample_weight=sample_weight)
			else:
				raise ValueError("Invalid option for smoother")
			per_feat_predictors.append(per_feat_predictor)
		agg_predictor = lambda tx: np.min([per_feat_predictors[i].predict(tx[:,1+i].reshape(-1,1)) for i in range(tx[:,1:].shape[1])], axis=0).reshape(-1,1)
		return agg_predictor, {}
	return interpretable_causal_random_forest_classifier_learner

def random_forest_classifier_aipw_transform(TX, Y, random_seed=None):
	T, X = TX[:,0], TX[:,1:]
	rf1 = RandomForestClassifier(random_state=random_seed).fit(X[T==1], Y[T==1])
	rf0 = RandomForestClassifier(random_state=random_seed).fit(X[T==0], Y[T==0])
	EY1, EY0 = rf1.predict(X), rf0.predict(X)
	prop = T.mean()
	pY1 = EY1 + (T / prop) * (Y - EY1)
	pY0 = EY0 + ((1 - T) / (1 - prop)) * (Y - EY0)
	pY = pY1 - pY0
	return pY

def make_interpretable_causal_isotonic_learner(random_seed=None):
	'''
	Assumes T is randomized and independent of X, so propensities are constant
	'''
	@ignore_warnings(category=UserWarning)
	@ignore_warnings(category=ConstantInputWarning)
	def interpretable_causal_isotonic_learner(TX, Y, sample_weight=None):
		pY = random_forest_classifier_aipw_transform(TX, Y, random_seed=random_seed)
		per_feat_predictors = []
		for i in range(TX[:,1:].shape[1]):
			feature = TX[:,1+i]
			per_feat_predictor = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(feature.reshape(-1,1), pY, sample_weight=sample_weight)
			per_feat_predictors.append(per_feat_predictor)
		agg_predictor = lambda tx: np.min([per_feat_predictors[i].predict(tx[:,1+i].reshape(-1,1)) for i in range(tx[:,1:].shape[1])], axis=0).reshape(-1,1)
		return agg_predictor, {}
	return interpretable_causal_isotonic_learner

def make_interpretable_facet_causal_random_forest_classifier_learner(mono=True, debias=False, min_box_cond_samps=100, random_seed=None):
	'''
	Slightly bespoke method. We require sample_weight to be passed and for sample_weight[i] > 1 to indicate points in the region.
	We otherwise ignore sample_weight when fitting the regressions.

	We either fit a unidirectional isotonic regression or a bidirectional isotonic regression.
	The bidirectional isotonic regression selects a few candidate modes and returns the best fit.
	'''
	base_learner = make_causal_random_forest_classifier_learner(random_seed)
	@ignore_warnings(category=UserWarning)
	@ignore_warnings(category=ConstantInputWarning)
	def interpretable_facet_causal_random_forest_classifier_learner(TX, Y, sample_weight=None):
		assert sample_weight is not None, "sample_weight must be provided"
		T, X = TX[:,0], TX[:,1:]
		# Calculate base predictions
		base_predictor, _ = base_learner(TX, Y)
		base_preds = base_predictor(TX).flatten()
		# Reverse engineer the in-region indicators
		in_reg = (sample_weight > 1)
		# Extract the boundary values for each facet. We approximate this by taking the coordinate-wise range of points in the region.
		facet_lower, facet_upper = X[in_reg].min(axis=0), X[in_reg].max(axis=0)
		coord_reg_dist = np.maximum((facet_lower - X) / X.std(axis=0), (X - facet_upper) / X.std(axis=0))
		coord_reg_dist = np.maximum(coord_reg_dist, 0)
		per_feat_predictors = []
		per_feat_predictions = []
		for i in range(X.shape[1]):
			# Get the relevant data that satisfies other box constraints (take all points in box, then also nearest points until we have min_box_cond_samps samples)
			keep_cols = np.ones(X.shape[1]).astype(bool)
			keep_cols[i] = False
			coord_total_dists = coord_reg_dist[:,keep_cols].mean(axis=1)
			box_cond_inds = (coord_total_dists == 0)
			if box_cond_inds.sum() < min_box_cond_samps:
				take_inds = np.argsort(coord_total_dists)[:min_box_cond_samps]
				box_cond_inds[take_inds] = True
			# Subset to relevant indices
			sub_feature = X[:,i][box_cond_inds]
			sub_base_preds = base_preds[box_cond_inds]
			# Fit univariate regression
			if mono:
				per_feat_predictor = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(sub_feature.reshape(-1,1), sub_base_preds)
				per_feat_predictor = lambda x, predictor=per_feat_predictor: predictor.predict(x)
			else:
				per_feat_predictor, _ = unimodal_fit(sub_feature, sub_base_preds)
			per_feat_predictors.append(per_feat_predictor)
			per_feat_predictions.append(per_feat_predictor(X[:,i]))
		per_feat_predictions = np.array(per_feat_predictions).T
		# Calculate an approximate shift term so that the predictions are not so negatively biased
		if debias:
			debias_shift = (per_feat_predictions.mean(axis=1) - per_feat_predictions.min(axis=1)).mean()
		else:
			debias_shift = 0
		agg_predictor = lambda tx: np.min([per_feat_predictors[i](tx[:,1+i]) for i in range(tx[:,1:].shape[1])], axis=0).reshape(-1,1) + debias_shift
		return agg_predictor, {}
	return interpretable_facet_causal_random_forest_classifier_learner

def make_nonneg_rct_causal_random_forest_learner(random_seed=None):
	'''
	This is a special class that should only be run with data from NonNegRCT because of the formatting
	'''
	def nonneg_rct_causal_random_forest_learner(YTX, pY, sample_weight=None):
		# Split Y and TX
		Y, TX = YTX[:,0], YTX[:,1:]
		# Split T and X
		T, X = TX[:,0], TX[:,1:]
		if np.isclose(np.var(T), 0):
			predictor = lambda ytx: np.zeros((ytx.shape[0], 1))
			is_const = True
			meta = {"is_const": True}
		else:
			if sample_weight is not None:
				sw1, sw0 = sample_weight[T==1], sample_weight[T==0]
			else:
				sw1, sw0 = None, None
			if np.isclose(Y[T==1].var(), 0):
				predictor_1 = lambda ytx: Y[T==1][0] * np.ones((len(ytx), 1))
			else:
				rf1 = RandomForestRegressor(max_depth=2, min_samples_leaf=10, max_features=None, random_state=random_seed).fit(X[T==1], Y[T==1], sample_weight=sw1)
				predictor_1 = lambda ytx: rf1.predict(ytx[:,2:])
			if np.isclose(Y[T==0].var(), 0):
				predictor_0 = lambda ytx: Y[T==0][0] * np.ones((len(ytx), 1))
			else:
				rf0 = RandomForestRegressor(max_depth=2, min_samples_leaf=10, max_features=None, random_state=random_seed).fit(X[T==0], Y[T==0], sample_weight=sw0)
				predictor_0 = lambda ytx: rf0.predict(ytx[:,2:])
			predictor = lambda ytx: predictor_1(ytx) - predictor_0(ytx)
			meta = {}
		return predictor, meta
	return nonneg_rct_causal_random_forest_learner

