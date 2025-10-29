import numpy as np
import scipy.stats

def equal_split_alpha(alpha, m):
	alpha_split = 1 - np.power(1 - alpha, 1 / m)
	return alpha_split

def get_remaining_alpha(protocol):
	obs_testing_history = protocol.get_observable_testing_history()
	return obs_testing_history.remaining_alpha.min()

class EqualAlphaStrategy:
	def __init__(self,
				 protocol,
				 test_thresh,
				 learner,
				 num_tests=20,
				 n_burn_in="auto",
				 n_min="auto",
				 reveal_batch_size=1,
				 refit_batch_size=1,
				 use_learner_weights=False,
				 tiebreak=False,
				 random_seed=None):
		# Save arguments
		self.protocol = protocol
		self.test_thresh = test_thresh
		self.learner = learner
		self.num_tests = num_tests
		self.n_burn_in = n_burn_in
		self.n_min = n_min
		self.reveal_batch_size = reveal_batch_size
		self.refit_batch_size = refit_batch_size
		self.tiebreak = tiebreak
		self.use_learner_weights = use_learner_weights
		self.random_seed = random_seed
		assert self.num_tests > 1, "num_tests must be at least 2"
		# Extract information
		self.alpha = self.protocol.alpha
		# Tracking metrics throughout strategy
		self.curr_strategy_phase = "burn_in"
		self.curr_n_pts_in_region = self.protocol.get_num_remaining_samples()
		self.sample_size_at_last_refit = None
		# Set defaults
		if self.n_burn_in == "auto":
			self.n_burn_in = max(30, int(self.curr_n_pts_in_region / self.num_tests))
		if self.n_min == "auto":
			self.n_min = max(30, int(0.05 * self.curr_n_pts_in_region))
		# Track revealed information
		self.revX = None
		self.revY = None
		self.meta_df = None
		# Track testing metrics
		self.n_tests_remaining = self.num_tests
		self.alpha_per_test = equal_split_alpha(self.alpha, self.num_tests)
		# Calculate the testing schedule; represents the number of tests we should have remaining at each sample size
		self.testing_schedule = {}
		for n in range(self.curr_n_pts_in_region, self.n_min, -1):
			w = (n - self.n_min - 1) / (self.curr_n_pts_in_region - self.n_min - 1)
			interp_num_tests = int(np.round((self.num_tests - 1) * w + 1 * (1 - w)))
			self.testing_schedule[n] = interp_num_tests
		self.testing_schedule[self.n_min] = 0
		# Initialize everything
		self._update_current_information()

	def _update_current_information(self):
		self.revX, self.revY, self.meta_df = self.protocol.get_current_information()
		self.curr_n_pts_in_region = self.protocol.get_num_remaining_samples()

	def _should_refit(self):
		if self.sample_size_at_last_refit is None:
			return True
		else:
			return (self.sample_size_at_last_refit - self.curr_n_pts_in_region) >= self.refit_batch_size

	def _fit_predictor(self):
		if self.use_learner_weights:
			# Calculate weights: samples outside region are given weight 1, samples inside region are given weight (n_masked + n_rev_inside) / n_rev_inside
			revX_registered, _, _ = self.protocol.get_current_information(cov_only=False)
			in_reg = self.protocol.region.in_region(revX_registered)
			sample_weight = np.ones(self.revX.shape[0])
			if in_reg.any():
				sample_weight[in_reg] = (self.n - self.revX.shape[0] + in_reg.sum()) / in_reg.sum()
			self.predictor, self.predictor_meta = self.learner(self.revX, self.revY, sample_weight)
		else:
			self.predictor, self.predictor_meta = self.learner(self.revX, self.revY)
		self.sample_size_at_last_refit = self.curr_n_pts_in_region

	def _update_tests(self):
		while self.n_tests_remaining > self.testing_schedule[self.curr_n_pts_in_region]:
			self.protocol.allocate_alpha_and_test(self.alpha_per_test)
			# Check for rejection
			if self.protocol.get_rejected_region() is not None:
				self.curr_strategy_phase = "finished"
				return
			self.n_tests_remaining -= 1

	def _increment_testing(self):
		if self.curr_strategy_phase != "testing":
			return
		if self.curr_n_pts_in_region < self.n_min:
			# Check if we have enough samples to do any testing; if not, we should quit
			self.curr_strategy_phase = "finished"
			return
		elif self.curr_n_pts_in_region == self.n_min:
			# If we are right at n_min, allocate all alpha and quit
			remaining_alpha = get_remaining_alpha(self.protocol)
			self.protocol.allocate_alpha_and_test(remaining_alpha)
			self.curr_strategy_phase = "finished"
			return
		# Check if any tests remain; if not, we should quit
		if self.n_tests_remaining == 0:
			self.curr_strategy_phase = "finished"
			return
		# Test
		self._update_tests()
		# Quit if we have a rejection
		if self.protocol.get_rejected_region() is not None:
			self.curr_strategy_phase = "finished"
			return
		# Quit if we don't have any alpha left
		if np.isclose(get_remaining_alpha(self.protocol), 0):
			self.curr_strategy_phase = "finished"
			return
		# Refit model
		if self._should_refit():
			self._fit_predictor()
		# Make sure batch size would not take us below n_min
		clipped_batch_size = min(self.curr_n_pts_in_region - self.n_min, self.reveal_batch_size)
		# Shrink region
		self.protocol.shrink_region(self.predictor, k_pts=clipped_batch_size, tiebreak=self.tiebreak)
		self._update_current_information()

	def run_burn_in(self):
		if self.curr_strategy_phase != "burn_in":
			return
		# Test
		self._update_tests()
		# Quit if we have a rejection
		if self.protocol.get_rejected_region() is not None:
			self.curr_strategy_phase = "finished"
			return
		# Burn-in
		self.protocol.reveal_random_points(k_pts=self.n_burn_in)
		# Update stage and current information
		self.curr_strategy_phase = "testing"
		self._update_current_information()

	def run_testing(self, verbose=False):
		while self.curr_strategy_phase == "testing":
			self._increment_testing()
			if verbose:
				print("n points remaining = {} / num tests remaining = {}".format(self.curr_n_pts_in_region, self.n_tests_remaining))

	def run_strategy(self, verbose=False):
		self.run_burn_in()
		self.run_testing(verbose=verbose)
