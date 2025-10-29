import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

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

# ------------------------------------ Search functions ------------------------------------ #

def search_for_target_specification_heterogeneous_linear_rct_size_only(d, s, rho, theta, target_size, n_reps=100000, random_seed=None):
	'''
	Selects tau so that P(E[Y | X] > 0) = target_size
	'''
	rng = np.random.RandomState(random_seed)
	# Construct beta
	beta = np.zeros(d)
	beta[:s] = 1.
	beta = theta * beta / np.linalg.norm(beta)
	# Generate X
	X = sample_ar_gaussian(n_reps, d, rho, rng)
	# Calculate tau required to shift 1 - target_size quantile to logit_mu
	unshifted_logits = X.dot(beta)
	unshifted_boundary = np.quantile(unshifted_logits, 1 - target_size)
	tau = -unshifted_boundary
	return tau

# ------------------------------------ Sampler class ------------------------------------ #

class HeterogeneousLinearRCT(DGP):
	'''
	Generative model: X_j ~ N(0, AR(rho))
					  CATE = tau + X^T beta
					  Y(0) ~ Expo(1) - 1
					  Y(1) = CATE + Y(0)
					  ||beta||_2 = theta
					  beta_{1:s} propto 1, all else = 0

					  T ~ Bern(0.5)
	'''
	def __init__(self, d, s, rho, theta, tau, ipw_transform=True, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.s = s
		self.rho = rho
		self.theta = theta
		self.tau = tau
		self.ipw_transform = ipw_transform
		# Construct betas
		self._construct_betas()
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def _construct_betas(self):
		self.beta = np.zeros(self.d)
		self.beta[:self.s] = 1.
		self.beta = self.theta * self.beta / np.linalg.norm(self.beta)

	def calculate_CATE(self, X):
		EYX = self.tau + X.dot(self.beta)
		return EYX

	def sample(self, n):
		# Generate X
		X = sample_ar_gaussian(n, self.d, self.rho, self.rng)
		# Calculate CATE
		CATE = self.calculate_CATE(X)
		# Generate Y
		Y0 = self.rng.exponential(size=n) - 1
		Y1 = CATE + Y0
		# Generate T
		T = self.rng.binomial(1, 0.5, size=n)
		# Format data
		Y = T * Y1 + (1 - T) * Y0
		TX = np.hstack((T.reshape(-1,1), X))
		if self.ipw_transform:
			pY = 2 * T * Y - 2 * (1 - T) * Y
			return X, pY
		else:
			return TX, Y

	def sample_noiseless(self, n):
		# Generate X
		X = sample_ar_gaussian(n, self.d, self.rho, self.rng)
		# Calculate CATE
		CATE = self.calculate_CATE(X)
		# Generate T
		T = self.rng.binomial(1, 0.5, size=n)
		if self.ipw_transform:
			return X, CATE
		else:
			return TX, CATE

	def get_optimal_region_metrics(self, n_reps=100000, recalculate=False):
		if (self.opt_utility_est is None) or recalculate:
			_, CATE = self.sample_noiseless(n_reps)
			self.opt_utility_est = (CATE * (CATE > 0)).mean()
			self.opt_utility_se = np.sqrt((CATE * (CATE > 0)).var() / n_reps)
			self.opt_reg_size = (CATE > 0).mean()
			self.opt_reg_size_se = np.sqrt((CATE > 0).var() / n_reps)
			opt_subgroup_CATEs = CATE[CATE > 0]
			if opt_subgroup_CATEs.shape[0] >= 2:
				self.opt_reg_ate = opt_subgroup_CATEs.mean()
				self.opt_reg_ate_se = np.sqrt(opt_subgroup_CATEs.var() / opt_subgroup_CATEs.shape[0])
			else:
				self.opt_reg_ate = np.nan
				self.opt_reg_ate_se = np.nan
		return self.opt_utility_est, self.opt_reg_size, self.opt_reg_ate, self.opt_utility_se, self.opt_reg_size_se, self.opt_reg_ate_se

	def estimate_region_metrics(self, region, n_reps=100000):
		if region is None:
			return 0, 0, np.nan, 0, 0, np.nan
		# Sample
		cov, CATE = self.sample_noiseless(n_reps)
		if not self.ipw_transform:
			cov = cov[:,1:]
		# Register units and check subgroup membership
		rs = self.rng.randint(0,2**32-1)
		unit_reg = UnitRegistrar(rs)
		regcov = unit_reg.register_units(cov)
		subgroup_inds = region.in_region(regcov)
		# Estimate region metrics
		util = (CATE * subgroup_inds).mean()
		util_se = np.sqrt((CATE * subgroup_inds).var() / n_reps)
		size = subgroup_inds.mean()
		size_se = np.sqrt(subgroup_inds.var() / n_reps)
		subgroup_CATEs = CATE[subgroup_inds]
		if subgroup_CATEs.shape[0] >= 2:
			sate = subgroup_CATEs.mean()
			sate_se = np.sqrt(subgroup_CATEs.var() / subgroup_CATEs.shape[0])
		else:
			sate = np.nan
			sate_se = np.nan
		return util, size, sate, util_se, size_se, sate_se

