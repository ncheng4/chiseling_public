import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

class KangSchaferRCT(DGP):
	'''
	Model from Kang and Schafer 2008

	Y(1) behaves according to the Kang and Schafer example but where we tune the intercept (0 by default) and scale up noise to sigma^2 (80 by default)
	Y(0) is a N(0, sigma^2)
	Propensities are 0.5
	'''
	def __init__(self, tau=0, sigma=80, ipw_transform=True, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.tau = tau
		self.sigma = sigma
		self.ipw_transform = ipw_transform
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def transform_Z(self, Z):
		X1 = np.exp(Z[:,0] / 2)
		X2 = Z[:,1] / (1 + np.exp(Z[:,0])) + 10
		X3 = np.power((Z[:,0] * Z[:,2]) / 25 + 0.6, 3)
		X4 = np.square(Z[:,1] + Z[:,3] + 20)
		X = np.array([X1, X2, X3, X4]).T
		return X

	def calculate_EY1(self, Z):
		beta = [27.4, 13.7, 13.7, 13.7]
		EY1 = self.tau + Z.dot(beta)
		return EY1

	def sample(self, n):
		# Generate latent Z
		Z = self.rng.normal(size=(n,4))
		# Calculate X
		X = self.transform_Z(Z)
		# Generate Y
		EY1 = self.calculate_EY1(Z)
		EY0 = 0
		eps = self.sigma * self.rng.normal(size=n)
		Y1 = EY1 + eps
		Y0 = EY0 + eps
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
		# Generate latent Z
		Z = self.rng.normal(size=(n,4))
		# Calculate X
		X = self.transform_Z(Z)
		# Generate Y
		EY1 = self.calculate_EY1(Z)
		EY0 = 0
		# Noiseless, so the outcome is the unobserved CATE for each unit
		CATE = EY1 - EY0
		# Generate T
		T = self.rng.binomial(1, 0.5, size=n)
		TX = np.hstack((T.reshape(-1,1), X))
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

