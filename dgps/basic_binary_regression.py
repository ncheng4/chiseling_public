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

def search_for_target_specification_binary_regression_size_only(d, s, rho, theta, mu, target_size, n_reps=100000, random_seed=None):
	'''
	Selects tau so that P(E[Y | X] > mu) = target_size
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
	logit_mu = scipy.special.logit(mu)
	tau = logit_mu - unshifted_boundary
	return tau

def search_for_target_specification_binary_regression(d, s, rho, mu, target_size, target_exp_t, theta_ub=40, theta_resolution=1000):
	'''
	Selects tau, theta so that P(E[Y | X] > mu) = target_size
	and sqrt(target_size) * (p - mu) / sqrt(p (1 - p)) is target_exp_t in expectation
	where p is mean in the region

	We solve this by noting that we have tau + X^T beta ~ N(tau, theta^2 * v) for some v
	So for any fixed theta, it is easy to find tau to match 1 - alpha quantile to logit(mu)
	Then we can calculate expected signal size and pick the theta that gets us closest to target_exp_t
	'''
	# Construct base beta
	beta = np.zeros(d)
	beta[:s] = 1.
	beta = beta / np.linalg.norm(beta)
	# Calculate variance of logits
	cov = AR1_covariance(d, rho)
	base_logit_var = beta.dot(cov).dot(beta)
	# Construct theta grid and get corresponding taus and calculate expected signal sizes
	theta_grid = np.linspace(0, theta_ub, theta_resolution)[1:]
	tau_grid = []
	sigsize_grid = []
	for theta in theta_grid:
		logit_var = np.square(theta) * base_logit_var
		# Calculate tau
		tau = scipy.special.logit(mu) - np.sqrt(logit_var) * scipy.stats.norm.ppf(1 - target_size)
		tau_grid.append(tau)
		# Calculate expected signal size
		region_mean = scipy.stats.norm.expect(scipy.special.expit, loc=tau, scale=np.sqrt(logit_var), lb=scipy.special.logit(mu), conditional=True)
		sigsize = np.sqrt(target_size) * (region_mean - mu) / np.sqrt(region_mean * (1 - region_mean))
		sigsize_grid.append(sigsize)
	tau_grid = np.array(tau_grid)
	sigsize_grid = np.array(sigsize_grid)
	# Pick the best setting
	best_ind = np.argmin(np.abs(sigsize_grid - target_exp_t))
	best_tau_theta = (tau_grid[best_ind], theta_grid[best_ind])
	best_sigsize_val = sigsize_grid[best_ind]
	return best_tau_theta, best_sigsize_val

# ------------------------------------ Sampler class ------------------------------------ #

class BasicBinaryRegression(DGP):
	'''
	Generative model: X_j ~ N(0, AR(rho))
					  logit(p) = tau + X^T beta
					  Y ~ Bern(p)
					  ||beta||_2 = theta
					  beta_{1:s} propto 1, all else = 0
	'''
	def __init__(self, d, s, rho, theta, tau, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.d = d
		self.s = s
		self.rho = rho
		self.theta = theta
		self.tau = tau
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

	def calculate_probs(self, X):
		logits = self.tau + X.dot(self.beta)
		probs = scipy.special.expit(logits)
		return probs

	def sample(self, n):
		# Generate X
		X = sample_ar_gaussian(n, self.d, self.rho, self.rng)
		# Calculate logits
		logits = self.tau + X.dot(self.beta)
		# Generate Y
		Y = self.rng.binomial(1, scipy.special.expit(logits))
		return X, Y

	def sample_noiseless(self, n):
		# Generate X
		X = sample_ar_gaussian(n, self.d, self.rho, self.rng)
		# Calculate logits
		logits = self.tau + X.dot(self.beta)
		# Calculate conditional mean
		EYX = scipy.special.expit(logits)
		return X, EYX

	def get_optimal_region_metrics(self, mu, n_reps=100000, recalculate=False):
		if (self.opt_utility_est is None) or recalculate:
			_, EYX = self.sample_noiseless(n_reps)
			self.opt_utility_est = ((EYX - mu) * (EYX > mu)).mean()
			self.opt_utility_se = np.sqrt(((EYX - mu) * (EYX > mu)).var() / n_reps)
			self.opt_reg_size = (EYX > mu).mean()
			self.opt_reg_size_se = np.sqrt((EYX > mu).var() / n_reps)
			opt_subgroup_scores = EYX[EYX > mu]
			if opt_subgroup_scores.shape[0] >= 2:
				self.opt_reg_mean = opt_subgroup_scores.mean()
				self.opt_reg_mean_se = np.sqrt(opt_subgroup_scores.var() / opt_subgroup_scores.shape[0])
			else:
				self.opt_reg_mean = np.nan
				self.opt_reg_mean_se = np.nan
		return self.opt_utility_est, self.opt_reg_size, self.opt_reg_mean, self.opt_utility_se, self.opt_reg_size_se, self.opt_reg_mean_se

	def estimate_region_metrics(self, mu, region, n_reps=100000):
		if region is None:
			return 0, 0, np.nan, 0, 0, np.nan
		# Sample
		cov, EYX = self.sample_noiseless(n_reps)
		# Register units and check subgroup membership
		rs = self.rng.randint(0,2**32-1)
		unit_reg = UnitRegistrar(rs)
		regcov = unit_reg.register_units(cov)
		subgroup_inds = region.in_region(regcov)
		# Estimate region metrics
		util = ((EYX - mu) * subgroup_inds).mean()
		util_se = np.sqrt(((EYX - mu) * subgroup_inds).var() / n_reps)
		size = subgroup_inds.mean()
		size_se = np.sqrt(subgroup_inds.var() / n_reps)
		subgroup_scores = EYX[subgroup_inds]
		if subgroup_scores.shape[0] >= 2:
			sub_mean = subgroup_scores.mean()
			sub_mean_se = np.sqrt(subgroup_scores.var() / subgroup_scores.shape[0])
		else:
			sub_mean = np.nan
			sub_mean_se = np.nan
		return util, size, sub_mean, util_se, size_se, sub_mean_se

