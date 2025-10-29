import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar


class LinearNonNegRCT(DGP):
	'''
	Generative model: CATE ~ tau * Bern(q)
					  X = CATE + Z_1, ..., CATE + Z_d where (Z_1,...,Z_d) are marginally N(0,1) but sum to 0 a.s.
					  Note that therefore CATE = sum(X) / d
					  Y(0) ~ Expo(1) - 1
					  Y(1) = CATE + Y(0)

					  T ~ Bern(0.5)
	'''
	def __init__(self, d, q, tau, tau0=0, ipw_transform=True, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.q = q
		self.tau = tau
		self.tau0 = tau0
		self.ipw_transform = ipw_transform
		# Pre-construct the fission moments
		self.fission_mean = np.zeros(self.d)
		self.fission_cov = np.ones((self.d, self.d)) * (-1 / (self.d - 1))
		np.fill_diagonal(self.fission_cov, 1)
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def sample(self, n):
		# Generate CATE
		eff_ind = self.rng.binomial(1, self.q, size=n)
		CATE = self.tau * eff_ind + self.tau0 * (1 - eff_ind)
		# Generate the fission variables
		Z = self.rng.multivariate_normal(self.fission_mean, self.fission_cov, size=n)
		# Calculate X
		X = CATE.reshape(-1,1) + Z
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
		# Generate CATE
		eff_ind = self.rng.binomial(1, self.q, size=n)
		CATE = self.tau * eff_ind + self.tau0 * (1 - eff_ind)
		# Generate the fission variables
		Z = self.rng.multivariate_normal(self.fission_mean, self.fission_cov, size=n)
		# Calculate X
		X = CATE.reshape(-1,1) + Z
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
