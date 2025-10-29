import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

class BasicLinearRCT(DGP):
	'''
	Generative model: X_j ~ N(0, I_d)
					  Y(1) = tau + X^T beta_prog + X^T beta_effect + eps, Y(0) = X^T beta_prog + eps, eps ~ N(0,1)
					  ||beta_prog||_2 = theta_prog, ||beta_effect||_2 = theta_effect
					  beta_prog_{1:s_prog} propto 1, beta_prog_{1:s_effect} propto 1, all else = 0

					  T ~ Bern(0.5), independently
	'''
	def __init__(self, d, s_prog, s_effect, theta_prog, theta_effect, tau, ipw_transform=True, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.s_prog = s_prog
		self.s_effect = s_effect
		self.theta_prog = theta_prog
		self.theta_effect = theta_effect
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
		self.beta_prog = np.zeros(self.d)
		self.beta_prog[:self.s_prog] = 1.
		self.beta_prog = self.theta_prog * self.beta_prog / np.linalg.norm(self.beta_prog)
		self.beta_effect = np.zeros(self.d)
		self.beta_effect[:self.s_effect] = 1.
		self.beta_effect = self.theta_effect * self.beta_effect / np.linalg.norm(self.beta_effect)

	def sample(self, n):
		# Generate X
		X = self.rng.normal(size=(n,self.d))
		# Generate Y
		eps = self.rng.normal(size=n)
		Y1 = self.tau + X.dot(self.beta_prog) + X.dot(self.beta_effect) + eps
		Y0 = X.dot(self.beta_prog) + eps
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
		X = self.rng.normal(size=(n,self.d))
		EY1 = self.tau + X.dot(self.beta_prog) + X.dot(self.beta_effect)
		EY0 = X.dot(self.beta_prog)
		T = self.rng.binomial(1, 0.5, size=n)
		# Noiseless, so the outcome is the unobserved CATE for each unit
		CATE = EY1 - EY0
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

