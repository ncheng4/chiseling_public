import numpy as np
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

class BinaryDesignRCT(DGP):
	'''
	Generative model: Dimension = d
					  X_{ij} ~ Bern(p) independently

					  CATE(X) = f(Xv) where v[:s] = 1 and rest are 0
					  Options for f:
					  	- linear: f(x) = tau + theta * x
					  	- step: f(x) = tau if x >= k and gamma otherwise
					  	- square: f(x) = tau + theta * x^2

					  Y(1) = CATE + epsilon
					  Y(0) = epsilon
					  epsilon ~ Expo(1) - 1

					  T ~ Bern(0.5), independently
	'''
	def __init__(self, d, p, s, k=None, tau=None, theta=None, gamma=None, f_opt=None, ipw_transform=True, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.p = p
		self.s = s
		self.k = k
		self.tau = tau
		self.theta = theta
		self.gamma = gamma
		self.f_opt = f_opt
		self.ipw_transform = ipw_transform
		# Construct f from option. Note that these allow vector x
		if self.f_opt == "linear":
			self.f = lambda x: self.tau + self.theta * x
		elif self.f_opt == "step":
			self.f = lambda x: self.tau * (x >= self.k) + self.gamma * (x < self.k)
		elif self.f_opt == "square":
			self.f = lambda x: self.tau + self.theta * np.square(x)
		else:
			raise ValueError("Invalid option for f_opt.")
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def calculate_CATE(self, X):
		Xv = X[:,:self.s].sum(axis=1)
		CATE = self.f(Xv)
		return CATE

	def sample(self, n):
		# Generate X
		X = self.rng.binomial(1, self.p, size=(n, self.d))
		# Calculate CATEs
		Xv = X[:,:self.s].sum(axis=1)
		CATE = self.f(Xv)
		# Generate Y
		Y0 = self.rng.exponential(size=n) - 1
		Y1 = Y0 + CATE
		# Generate T
		T = self.rng.binomial(1, 0.5, size=n)
		# Format data
		Y = T * Y1 + (1 - T) * Y0
		if self.ipw_transform:
			pY = 2 * T * Y - 2 * (1 - T) * Y
			return X, pY
		else:
			TX = np.hstack((T.reshape(-1,1), X))
			return TX, Y

	def sample_noiseless(self, n):
		# Generate X
		X = self.rng.binomial(1, self.p, size=(n, self.d))
		# Calculate CATEs
		Xv = X[:,:self.s].sum(axis=1)
		CATE = self.f(Xv)
		# Noiseless, so the outcome is the unobserved CATE for each unit
		if self.ipw_transform:
			return X, CATE
		else:
			# Generate T
			T = self.rng.binomial(1, 0.5, size=n)
			TX = np.hstack((T.reshape(-1,1), X))
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

