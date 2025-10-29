import numpy as np
import scipy.stats

class SimpleMarginStrategy:
	def __init__(self,
				 protocol,
				 test_thresh,
				 learner,
				 n_burn_in,
				 alpha_init=0,
				 refit_batch_prop=0.05,
				 reveal_batch_prop=0.01,
				 margin_width=1,
				 n_min='auto',
				 alpha_min='auto',
				 use_learner_weights=False,
				 skip_const_predictor=False,
				 shrink_to_boundary=True,
				 tiebreak=False):
		# Save arguments
		self.protocol = protocol
		self.test_thresh = test_thresh
		self.learner = learner
		self.n_burn_in = n_burn_in
		self.alpha_init = alpha_init
		self.refit_batch_prop = refit_batch_prop
		self.reveal_batch_prop = reveal_batch_prop
		self.margin_width = margin_width
		self.n_min = n_min
		self.alpha_min = alpha_min
		self.use_learner_weights = use_learner_weights
		self.skip_const_predictor = skip_const_predictor
		self.shrink_to_boundary = shrink_to_boundary
		self.tiebreak = tiebreak
		# Extract information
		self.n = self.protocol.Y.shape[0]
		self.alpha = self.protocol.alpha
		# Set defaults
		if 0 < self.n_burn_in < 1:
			self.n_burn_in = max(1, int(self.n * self.n_burn_in))
		if isinstance(self.n_min, str) and self.n_min == 'auto':
			self.n_min = int(max(30, 0.05 * self.n))
		if isinstance(self.alpha_min, str) and self.alpha_min == 'auto':
			self.alpha_min = 1 - np.power(1 - self.alpha, 1/40)
		# More arguments
		self.refit_batch_size = max(1, int(self.n * self.refit_batch_prop))
		self.reveal_batch_size = None
		# Tracking metrics throughout strategy
		self.curr_strategy_phase = "initialize"
		self.curr_n_pts_in_region = None
		self.sample_size_at_last_refit = None
		self.n_points_at_boundary = None
		# Track revealed information and current predictor
		self.revX = None
		self.revY = None
		self.meta_df = None
		self.remaining_alpha = None
		self.spent_alpha = None
		self.predictor = None
		self.predictor_meta = None
		# Metrics
		self.metrics = {}
		# Initialize everything
		self._update_current_information()
		self._update_alpha_spending()

	def _update_current_information(self):
		self.revX, self.revY, self.meta_df = self.protocol.get_current_information()
		self.curr_n_pts_in_region = self.protocol.get_num_remaining_samples()

	def _update_alpha_spending(self):
		self.remaining_alpha = self.protocol.get_observable_testing_history().remaining_alpha.min()
		self.spent_alpha = 1 - (1 - self.alpha) / (1 - self.remaining_alpha)

	def _calculate_alpha_budget(self):
		# If at the very beginning, we spend alpha_init
		if self.curr_strategy_phase == "initialize":
			return self.alpha_init
		# Otherwise, we only define the alpha budget if we have already hit the boundary
		assert self.n_points_at_boundary is not None, "This function may only be called once we have logged the number of samples upon chiseling to the boundary."
		if self.curr_n_pts_in_region == self.n_min:
			return self.alpha
		elif self.margin_width == 0:
			return self.alpha
		else:
			# We interpolate linearly between alpha_init and alpha based on how far through the margin we've gone
			allocation_width = self.margin_width * (self.n_points_at_boundary - self.n_min)
			allocation_x = (self.n_points_at_boundary - self.curr_n_pts_in_region) / allocation_width
			# Clip to be less than 1
			allocation_x = min(1, allocation_x)
			return self.alpha_init + allocation_x * (self.alpha - self.alpha_init)

	def _spend_up_to_alpha_budget(self):
		alpha_budget = self._calculate_alpha_budget()
		alloc_alpha = 1 - (1 - alpha_budget) / (1 - self.spent_alpha)
		if alloc_alpha >= self.alpha_min:
			self.protocol.allocate_alpha_and_test(alloc_alpha)
			self._update_alpha_spending()
			rejected = (self.protocol.get_rejected_region() is not None)
			return rejected
		else:
			return False

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
		# Save the sample size at which we refit, resetting the cycle
		self.sample_size_at_last_refit = self.curr_n_pts_in_region

	def _should_refit(self):
		if self.sample_size_at_last_refit is None:
			return True
		else:
			return (self.sample_size_at_last_refit - self.curr_n_pts_in_region) >= self.refit_batch_size

	def _adaptively_shrink_to_boundary(self):
		below_boundary = True
		while below_boundary:
			# Make sure we can shrink without going below n_min
			if self.curr_n_pts_in_region <= self.n_min:
				break
			# Make sure batch size would not take us below n_min
			clipped_batch_size = min(self.curr_n_pts_in_region - self.n_min, self.refit_batch_size)
			# Fit predictor
			self._fit_predictor()
			# Exit if predictor is constant, if setting is toggled
			if self.skip_const_predictor and "is_const" in self.predictor_meta.keys() and self.predictor_meta["is_const"]:
				break
			# Shrink region up to boundary
			recent_revX, recent_revY, recent_thresh = self.protocol.shrink_region(self.predictor,
																				  k_pts=clipped_batch_size,
																				  max_thresh=np.array([self.test_thresh]),
																				  tiebreak=self.tiebreak)
			self._update_current_information()
			n_revealed = recent_revY.shape[0]
			if n_revealed < clipped_batch_size:
				below_boundary = False

	def _shrink_region(self, batch_size):
		if batch_size > 0:
			# Refit predictor
			if self._should_refit():
				self._fit_predictor()
			if self.skip_const_predictor and "is_const" in self.predictor_meta.keys() and self.predictor_meta["is_const"]:
				# If predictor is constant, we just reveal points randomly
				self.protocol.reveal_random_points(k_pts=batch_size)
			else:
				# Shrink with no limiter
				self.protocol.shrink_region(self.predictor, k_pts=batch_size, tiebreak=self.tiebreak)
			# Update current information
			self._update_current_information()

	def run_initialize(self, verbose=False):
		if self.curr_strategy_phase != "initialize":
			return
		# Run the initial test
		rejected = self._spend_up_to_alpha_budget()
		# Quit if we reach a rejection
		if rejected:
			self.curr_strategy_phase = "finished"
			return
		# Reveal random points
		self.protocol.reveal_random_points(k_pts=self.n_burn_in)
		self._update_current_information()
		# Shrink to boundary
		if self.shrink_to_boundary:
			self._adaptively_shrink_to_boundary()
		# Save the number of points left after shrinking to boundary
		self.n_points_at_boundary = self.curr_n_pts_in_region
		# Calculate the reveal batch size based on this
		n_testable_pts = self.n_points_at_boundary - self.n_min
		self.reveal_batch_size = max(1, int(self.reveal_batch_prop * n_testable_pts))
		# Update strategy phase
		self.curr_strategy_phase = "main"

	def _increment_main(self):
		if self.curr_strategy_phase != "main":
			return
		elif self.curr_n_pts_in_region < self.n_min:
			# No samples remain, we terminate
			self.curr_strategy_phase = "finished"
			return
		elif self.curr_n_pts_in_region == self.n_min:
			# If we are at n_min, spend all remaining alpha if possible and terminate
			self._spend_up_to_alpha_budget()
			self.curr_strategy_phase = "finished"
			return
		# Test up to the current alpha allocation
		rejected = self._spend_up_to_alpha_budget()
		# Quit if we reach a rejection
		if rejected:
			self.curr_strategy_phase = "finished"
			return
		# Quit if no alpha remains
		if np.isclose(self.remaining_alpha, 0):
			# No alpha remaining, we terminate
			self.curr_strategy_phase = "finished"
			return
		# Make sure batch size would not take us below n_min
		clipped_batch_size = min(self.curr_n_pts_in_region - self.n_min, min(self.reveal_batch_size, self.refit_batch_size))
		# Shrink region
		self._shrink_region(clipped_batch_size)
		# Save metrics
		curr_region_mass = self.protocol.protocol_metadata.reg_mass_est.min()
		self.metrics[self.curr_n_pts_in_region] = {"spent_alpha": self.spent_alpha,
												   "remaining_alpha": self.remaining_alpha,
												   "n_left_in_region": self.curr_n_pts_in_region,
												   "region_mass_estimate": curr_region_mass}

	def run_main(self, verbose=False):
		while self.curr_strategy_phase == "main":
			self._increment_main()
			if verbose and self.curr_strategy_phase != "finished":
				last_key = min(self.metrics.keys())
				print("METRICS =", self.metrics[last_key])
		# Save last metrics
		rejected = (self.protocol.get_rejected_region() is not None)
		curr_region_mass = self.protocol.protocol_metadata.reg_mass_est.min()
		self.metrics["FINAL"] = {"rejected": rejected,
								 "spent_alpha": self.spent_alpha,
								 "remaining_alpha": self.remaining_alpha,
								 "n_left_in_region": self.curr_n_pts_in_region,
								 "region_mass_estimate": curr_region_mass}
		if verbose:
			print("SUMMARY: rejected = {}, spent_alpha = {}, remaining_alpha = {}, n_left_in_region = {}, region_mass_estimate = {}".format(self.metrics["FINAL"]["rejected"],
																													  						self.metrics["FINAL"]["spent_alpha"],
																													  						self.metrics["FINAL"]["remaining_alpha"],
																													  						self.metrics["FINAL"]["n_left_in_region"],
																													  						self.metrics["FINAL"]["region_mass_estimate"]))

	def run_strategy(self, verbose=False):
		self.run_initialize(verbose=verbose)
		self.run_main(verbose=verbose)
