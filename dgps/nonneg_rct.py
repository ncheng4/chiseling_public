import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar
from ..source.protocol.utils import aipw_intercept_pseudo_outcome


class NonNegRCT(DGP):
	'''
	Generative model: R_eff = {x : x1,...,xs > 0} intersect [-1,1]^d
					  R_null = R_eff^c intersect [-1,1]^d
					  X | {X in R_eff} and X | {X in R_null} uniform over support
					  P(X in R_eff) = q
					  CATE(x) = tau for x in R_eff, tau0 otherwise (tau0 = 0 by default)
					  Y(0) ~ Expo(1) - 1
					  Y(1) = CATE + Y(0)

					  T ~ Bern(0.5)

					  By default, we return the AIPW transform as Y, and but the raw Y as the first covariate, T as the second covariate,
					  and the rest is X. To make this work with the Benchmark module, I should use a special learner that recognizes Y
					  as the first covariate and ignores the Y passed by the strategy class.
	'''
	def __init__(self, d, s, q, tau, tau0=0, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.s = s
		self.q = q
		self.tau = tau
		self.tau0 = tau0
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def calculate_CATE(self, X):
		in_R_eff = (X[:,:self.s] > 0).all(axis=1)
		CATE = self.tau * in_R_eff + self.tau0 * (1 - in_R_eff)
		return CATE

	def sample_X(self, n):
		in_R_eff = self.rng.binomial(1, self.q, size=n)
		X_R_eff = np.hstack((self.rng.uniform(0,1,size=(n,self.s)), self.rng.uniform(-1,1,size=(n,self.d-self.s))))
		X_R_null = self.rng.uniform(-1,1,size=(n,self.d))
		nonnull = (X_R_null[:,:self.s] > 0).all(axis=1)
		while nonnull.any():
			X_R_null[nonnull] = self.rng.uniform(-1,1,size=(nonnull.sum(),self.d))
			nonnull = (X_R_null[:,:self.s] > 0).all(axis=1)
		X = (in_R_eff.reshape(-1,1) * X_R_eff) + ((1 - in_R_eff).reshape(-1,1) * X_R_null)
		return X

	def sample(self, n):
		# Generate X
		X = self.sample_X(n)
		# Calculate CATE
		CATE = self.calculate_CATE(X)
		# Generate Y
		Y0 = self.rng.exponential(size=n) - 1
		Y1 = CATE + Y0
		# Generate T
		T = self.rng.binomial(1, 0.5, size=n)
		# Format data
		Y = T * Y1 + (1 - T) * Y0
		# AIPW transform
		pY = aipw_intercept_pseudo_outcome(T, X, Y, propensities=0.5, cv=5, random_seed=0)
		# Format covariates
		YTX = np.hstack((Y.reshape(-1,1), T.reshape(-1,1), X))
		return YTX, pY

	def sample_noiseless(self, n):
		# Generate X
		X = self.sample_X(n)
		# Calculate CATE
		CATE = self.calculate_CATE(X)
		# Generate Y
		Y0 = self.rng.exponential(size=n) - 1
		Y1 = CATE + Y0
		# Generate T
		T = self.rng.binomial(1, 0.5, size=n)
		# Format data
		Y = T * Y1 + (1 - T) * Y0
		# AIPW transform
		pY = aipw_intercept_pseudo_outcome(T, X, Y, propensities=0.5, cv=5, random_seed=0)
		# Format covariates
		YTX = np.hstack((Y.reshape(-1,1), T.reshape(-1,1), X))
		return YTX, CATE

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

