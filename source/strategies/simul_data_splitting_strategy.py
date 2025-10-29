import numpy as np
import scipy.stats

from sklearn.model_selection import train_test_split
from ..protocol.IRST import Region, PrioritizationScore

# ---------------------------- Helper functions ---------------------------- #

def construct_subgroup_indics(scores, thresh, n_subgroups, n_min):
	'''
	Construct n_subgroups number of nested subgroups beginning with {scores > thresh}, equally spaced in quantiles, and having no fewer than n_min samples
	Subgroups are allowed to be duplicated/identical
	Returns a matrix of subgroup indicators of shape (len(scores), n_subgroups) and a vector of cutoffs used to define the subgroups
	If number of elements i with scores[i] > thresh is less than n_min, return empy matrix and vector (of shape (len(scores), 0) and 0)
	'''
	sorted_pos_scores = np.sort(scores[scores > thresh])
	if len(sorted_pos_scores) < n_min:
		return np.zeros(shape=(len(scores), 0)), np.zeros(shape=0)
	elif n_subgroups == 1:
		return scores.reshape(-1,1) > thresh, np.array([thresh])
	else:
		satisfying_thresh_inds = [i for i in range(len(sorted_pos_scores)) if (sorted_pos_scores > sorted_pos_scores[i]).sum() >= n_min]
		if len(satisfying_thresh_inds) == 0:
			return scores.reshape(-1,1) > thresh, np.array([thresh])
		else:
			last_thresh_ind = max(satisfying_thresh_inds)
			all_thresh_inds = np.array([last_thresh_ind * (i / (n_subgroups - 1)) for i in range(1, n_subgroups)]).astype(int)
			all_thresh = sorted_pos_scores[all_thresh_inds]
			all_thresh = np.hstack(([thresh], all_thresh))
			subgroup_indics = (scores.reshape(-1,1) > all_thresh)
			return subgroup_indics, all_thresh

def subgroup_pt_est(Y, subgroup_inds):
	# Safe divide
	pt_est = Y.dot(subgroup_inds) / np.maximum(subgroup_inds.sum(axis=0), 1)
	pt_est[subgroup_inds.sum(axis=0) == 0] = np.nan
	return pt_est

def simultaneous_subgroup_lcb(Y, subgroup_inds, alpha, dropna=True, n_boots=500, random_seed=None):
	# Return nothing if no subgroups
	if subgroup_inds.shape[1] == 0:
		return np.array([])
	rng = np.random.RandomState(random_seed)
	# Calculate point estimate
	pt_est = subgroup_pt_est(Y, subgroup_inds)
	# Calculate bootstrap point estimates
	all_boot_pt_est = []
	for _ in range(n_boots):
		boot_inds = rng.choice(Y.shape[0], Y.shape[0], replace=True)
		boot_pt_est = subgroup_pt_est(Y[boot_inds], subgroup_inds[boot_inds])
		if dropna and np.isnan(boot_pt_est).any():
			continue
		all_boot_pt_est.append(boot_pt_est)
	all_boot_pt_est = np.array(all_boot_pt_est)
	# Return trivial LCB if no bootstrap estimates remain
	if len(all_boot_pt_est) == 0:
		return np.array([-np.inf] * subgroup_inds.shape[1])
	# Calculate marginal estimand variances
	sigmas = all_boot_pt_est.std(axis=0)
	# If sigmas[i] = 0, we drop i from the max stats calculation
	sub_nonzero_sigma_inds = ~np.isclose(sigmas, 0)
	if sub_nonzero_sigma_inds.any():
		# Calculate max stats on nonzero sigmas
		sub_max_stats = ((all_boot_pt_est[:,sub_nonzero_sigma_inds] - pt_est[sub_nonzero_sigma_inds]) / sigmas[sub_nonzero_sigma_inds]).max(axis=1)
		# Quantile of test statistic
		q = np.quantile(sub_max_stats, 1 - alpha)
		# Construct lower confidence bound
		lcb = sub_nonzero_sigma_inds * (pt_est - sigmas * q) + (1 - sub_nonzero_sigma_inds) * pt_est
		return lcb
	else:
		return pt_est

def binary_test_stat(Y, subgroup_inds, test_thresh):
	pt_ests = subgroup_pt_est(Y, subgroup_inds)
	counts = subgroup_inds.sum(axis=0)
	test_stats = []
	for i in range(subgroup_inds.shape[1]):
		if (counts[i] == 0) or (pt_ests[i] <= test_thresh):
			test_stats.append(0)
		elif pt_ests[i] == 1:
			test_stats.append(np.inf)
		else:
			test_stats.append(np.sqrt(counts[i] / (pt_ests[i] * (1 - pt_ests[i]))) * (pt_ests[i] - test_thresh))
	test_stats = np.array(test_stats)
	return test_stats

def approx_simultaneous_subgroup_test_binary(Y, subgroup_inds, alpha, test_thresh, n_sims=500, randomize=True, random_seed=None):
	# Return nothing if no subgroups
	if subgroup_inds.shape[1] == 0:
		return np.array([])
	# Check binary
	assert set(np.unique(Y)).issubset([0,1]), "Y must be binary"
	rng = np.random.RandomState(random_seed)
	# Calculate test statistics
	test_stats = binary_test_stat(Y, subgroup_inds, test_thresh)
	# Calculate null test statistics
	all_null_test_stats = []
	for _ in range(n_sims):
		Y_null = rng.binomial(1, test_thresh, size=len(Y))
		null_test_stats = binary_test_stat(Y_null, subgroup_inds, test_thresh)
		all_null_test_stats.append(null_test_stats)
	all_null_test_stats = np.array(all_null_test_stats)
	# Calculate max statistics
	null_max_stats = all_null_test_stats.max(axis=1)
	if randomize:
		# Augment max statistics and test statistic with uniform
		aug_null_max_stats = np.vstack((null_max_stats, rng.uniform(size=len(null_max_stats)))).T
		test_aug = rng.uniform()
		# Calculate rejections
		rejections = []
		for t in test_stats:
			num_exceed = ((aug_null_max_stats[:,0] > t) | ((aug_null_max_stats[:,0] == t) & (aug_null_max_stats[:,1] >= test_aug))).sum()
			pval = (1 + num_exceed) / (1 + len(aug_null_max_stats))
			rejections.append(pval <= alpha)
		return np.array(rejections)
	else:
		rejections = []
		for t in test_stats:
			num_exceed = (null_max_stats >= t).sum()
			pval = (1 + num_exceed) / (1 + len(null_max_stats))
			rejections.append(pval <= alpha)
		return np.array(rejections)



# ---------------------------- Strategy class ---------------------------- #

class SimulDataSplittingStrategy:
	'''
	If binary = True, we run an approximate test that just simulates the distribution of the test statistic at the boundary of the global null
	'''
	def __init__(self,
				 X,
				 Y,
				 train_ratio,
				 learner,
				 alpha,
				 test_thresh,
				 n_regs=10,
				 pY=None,
				 n_min=30,
				 binary=False,
				 randomize=True,
				 random_seed=None):
		self.X = X
		self.Y = Y
		self.train_ratio = train_ratio
		self.learner = learner
		self.alpha = alpha
		self.test_thresh = test_thresh
		self.n_regs = n_regs
		self.pY = pY
		self.n_min = n_min
		self.binary = binary
		self.randomize = randomize
		self.random_seed = random_seed
		# Set defaults
		if self.pY is None:
			self.pY = Y
		# Strategy variables
		self.X_train = None
		self.Y_train = None
		self.X_test = None
		self.Y_test = None
		self.pY_train = None
		self.pY_test = None
		self.predictor = None
		self.predictor_meta = None
		self.test_cutoffs = None
		self.test_subgroup_inds = None
		self.lcb = None
		self.rejected_indics = None
		self.rej_cutoff = None
		# Result variables
		self.region = None
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
		# Construct subgroup indicators and cutoffs on test
		test_scores = self.predictor(self.X_test).flatten()
		self.test_subgroup_inds, self.test_cutoffs = construct_subgroup_indics(test_scores, self.test_thresh, self.n_regs, self.n_min)
		# Run test
		if self.binary:
			self.rejected_indics = approx_simultaneous_subgroup_test_binary(self.pY_test,
																			self.test_subgroup_inds,
																			self.alpha,
																			test_thresh=self.test_thresh,
																			randomize=self.randomize,
																			random_seed=self.random_seed)
		else:
			self.lcb = simultaneous_subgroup_lcb(self.pY_test, self.test_subgroup_inds, self.alpha, random_seed=self.random_seed)
			self.rejected_indics = (self.lcb > self.test_thresh)
		# Get largest rejection
		if self.rejected_indics.any():
			self.rejected = True
			# Get the cutoff for the first significant region
			rej_region_ind = min(np.where(self.rejected_indics)[0])
			self.rej_cutoff = self.test_cutoffs[rej_region_ind]
			# Construct the region
			self.region = Region()
			pscore = PrioritizationScore(predictor=self.predictor)
			self.region.add_region_constraint("learner", pscore, [self.rej_cutoff], metadata={})
		else:
			self.rejected = False
