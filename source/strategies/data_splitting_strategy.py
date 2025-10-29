import numpy as np
import scipy.stats

from sklearn.model_selection import train_test_split
from ..protocol.IRST import Region, PrioritizationScore

def binom_quant(quantile, mu, sample_size, rng=None):
	'''
	Randomized exact binomial quantile
	'''
	if quantile >= 1:
		return np.inf
	# Calculate the CDF; note for convenience we start it at -1
	z_space = np.arange(-1, sample_size + 1)
	binom_cdf = scipy.stats.binom.cdf(z_space, sample_size, mu)
	# Calculate the lower and upper quantiles and associated probabilities
	z_upper_ind = np.where(binom_cdf > quantile)[0].min()
	z_lower_ind = np.where(binom_cdf <= quantile)[0].max()
	z_upper = z_space[z_upper_ind]
	z_lower = z_space[z_lower_ind]
	upper_cdf = binom_cdf[z_upper_ind]
	lower_cdf = binom_cdf[z_lower_ind]
	# Calculate probability of sampling z_upper
	z_upper_prob = (quantile - lower_cdf) / (upper_cdf - lower_cdf)
	# Sample quantile
	if rng is None:
		sel = np.random.binomial(1, z_upper_prob)
	else:
		sel = rng.binomial(1, z_upper_prob)
	z = z_upper * sel + z_lower * (1 - sel)
	return z

class DataSplittingStrategy:
	def __init__(self,
				 X,
				 Y,
				 train_ratio,
				 learner,
				 alpha,
				 test_thresh,
				 pY=None,
				 binary=False,
				 randomize=True,
				 n_min=2,
				 random_seed=None):
		self.X = X
		self.Y = Y
		self.train_ratio = train_ratio
		self.learner = learner
		self.alpha = alpha
		self.test_thresh = test_thresh
		self.pY = pY
		self.binary = binary
		self.randomize = randomize
		self.n_min = n_min
		self.random_seed = random_seed
		# Set defaults
		if self.pY is None:
			self.pY = Y
		self.rng = np.random.RandomState(random_seed)
		# Strategy variables
		self.X_train = None
		self.Y_train = None
		self.X_test = None
		self.Y_test = None
		self.predictor = None
		self.predictor_meta = None
		self.test_subgroup_inds = None
		# Result variables
		self.region = None
		self.rand_critval = None
		self.teststat = None
		self.pval = None
		self.rejected = None

	def run_strategy(self):
		# Split data
		self.X_train, self.X_test, self.Y_train, self.Y_test, self.pY_train, self.pY_test = train_test_split(self.X,
																											 self.Y,
																											 self.pY,
																											 train_size=self.train_ratio,
																											 random_state=self.random_seed)
		# Estimate and construct region
		self.predictor, self.predictor_meta = self.learner(self.X_train, self.Y_train)
		self.region = Region()
		pscore = PrioritizationScore(predictor=self.predictor)
		self.region.add_region_constraint("learner", pscore, [self.test_thresh], metadata={})
		# Evaluate region
		self.test_subgroup_inds = (self.predictor(self.X_test) > self.test_thresh).flatten()
		if self.test_subgroup_inds.sum() < self.n_min:
			if self.randomize:
				self.rejected = bool(np.random.binomial(1, self.alpha))
			else:
				self.rejected = False
		else:
			pY_test_subgroup = self.pY_test[self.test_subgroup_inds]
			if self.binary:
				if self.randomize:
					self.rand_critval = binom_quant(1 - self.alpha, self.test_thresh, pY_test_subgroup.shape[0], rng=self.rng)
					self.rejected = (pY_test_subgroup.sum() > self.rand_critval)
				else:
					res = scipy.stats.binomtest(pY_test_subgroup.sum(), pY_test_subgroup.shape[0], p=self.test_thresh, alternative="greater")
					self.teststat, self.pval = res.statistic, res.pvalue
					self.rejected = (self.pval <= self.alpha)
			else:
				res = scipy.stats.ttest_1samp(pY_test_subgroup, popmean=self.test_thresh, alternative="greater")
				self.teststat, self.pval = res.statistic, res.pvalue
				self.rejected = (self.pval <= self.alpha)

