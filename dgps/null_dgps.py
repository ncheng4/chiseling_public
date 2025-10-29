import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# ------------------------------------ Helper samplers ------------------------------------ #

def AR1_covariance(d, rho):
	inds = np.arange(d)
	dist = np.abs(inds - inds.reshape(-1,1))
	cov = np.power(rho, dist)
	return cov

def sample_ar_gaussian(n, d, ar1_rho, rng):
	cov = AR1_covariance(d, ar1_rho)
	X = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
	return X

def null_dgps_aipw_transform(T, X, Y, cv=5, first_k_covs=5, random_seed=None):
	'''
	Implement AIPW transform using linear regression with first 5 covariates and 5-fold cross-fitting
	'''
	# We know propensities = 0.5 in this setting
	propensities = 0.5 * np.ones(len(Y))
	# Initialize
	pY1 = np.zeros(len(Y))
	pY0 = np.zeros(len(Y))
	# Split
	kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
	for i, (train_index, test_index) in enumerate(kf.split(X)):
		# Split
		X_train, T_train, Y_train = X[train_index], T[train_index], Y[train_index]
		X_test, T_test, Y_test = X[test_index], T[test_index], Y[test_index]
		# Run regressions
		if (T_train == 1).sum() > 1:
			linreg1 = LinearRegression().fit(X_train[T_train == 1][:,:first_k_covs], Y_train[T_train == 1])
			m1 = linreg1.predict(X_test[:,:first_k_covs])
		else:
			m1 = 0
		if (T_train == 0).sum() > 1:
			linreg0 = LinearRegression().fit(X_train[T_train == 0][:,:first_k_covs], Y_train[T_train == 0])
			m0 = linreg0.predict(X_test[:,:first_k_covs])
		else:
			m0 = 0
		# Calculate AIPW estimate on test samples
		pY1[test_index] = m1 + T[test_index] * (Y[test_index] - m1) / propensities[test_index]
		pY0[test_index] = m0 + (1 - T[test_index]) * (Y[test_index] - m0) / (1 - propensities[test_index])
	# Combine
	pY = pY1 - pY0
	return pY

# ------------------------------------ Sampler class ------------------------------------ #

class NullDGPs(DGP):
	'''
	cov_type:
		- corrnorm: X ~ normal with AR(0.2) covariance
		- binary: X ~ independent Rademacher
		- expo: X ~ independent Expo(1) - 1

	If binary, err_type ignored, and Y ~ Bern(0.5)

	Otherwise, Y = TY(1) + (1 - T)Y(0) where T ~ Bern(0.5), Y(1) = Y(0) = f(x) + eps and f(x) = arctan((x_1 + ... + x_5) / sqrt(5))
	err_type:
		- expo: eps ~ Expo(1) - 1
		- t5: eps ~ t-distribution with 5 degrees of freedom
		- hetnorm: eps ~ N(0, 1 + f(x)^2)

	If binary, return (X, Y)
	Otherwise, return (TX, Y); transformations must be dealt with downstream
	'''
	def __init__(self, d, binary=None, cov_type=None, err_type=None, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.cov_type = cov_type
		self.binary = binary
		self.err_type = err_type
		# Check arguments
		assert self.binary in [True, False], "Invalid option for binary"
		assert self.cov_type in ["corrnorm", "binary", "expo"], "Invalid cov_type"
		assert self.binary or self.err_type in ["expo", "t5", "hetnorm"], "If binary = False, err_type must belong to {expo, t5, hetnorm}"
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def sample_X(self, n):
		if self.cov_type == "corrnorm":
			X = sample_ar_gaussian(n, self.d, 0.2, self.rng)
		elif self.cov_type == "binary":
			X = 2 * self.rng.binomial(1, 0.5, size=(n, self.d)) - 1
		elif self.cov_type == "expo":
			X = self.rng.exponential(size=(n, self.d)) - 1
		return X

	def calculate_prognostic(self, X):
		fx = np.arctan(X[:,:5].sum() / np.sqrt(5))
		return fx

	def sample_error(self, X):
		if self.err_type == "expo":
			eps = self.rng.exponential(size=len(X)) - 1
		elif self.err_type == "t5":
			eps = self.rng.standard_t(df=5, size=len(X))
		elif self.err_type == "hetnorm":
			fx = self.calculate_prognostic(X)
			eps = np.sqrt(1 + np.power(fx, 2)) * self.rng.normal(size=len(X))
		return eps

	def sample(self, n):
		X = self.sample_X(n)
		if self.binary:
			Y = self.rng.binomial(1, 0.5, size=n)
			return X, Y
		else:
			# Sample treatment indicators
			T = self.rng.binomial(1, 0.5, size=n)
			# Sample potential outcomes
			fx = self.calculate_prognostic(X)
			Y0 = fx + self.sample_error(X)
			Y1 = Y0
			# Construct observed outcomes
			Y = T * Y1 + (1 - T) * Y0
			# Construct augmented design
			TX = np.hstack((T.reshape(-1,1), X))
			return TX, Y

	def estimate_region_metrics(self, region):
		return 0, 0, 0, 0, 0, 0
