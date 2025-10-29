import numpy as np
import scipy.stats

class AlphaSpendingStrategy:
	def __init__(self,
				 protocol,
				 test_thresh,
				 learner,
				 n_burn_in,
				 reveal_batch_size=1,
				 refit_batch_size=1,
				 n_min='auto',
				 alpha_min='auto',
				 alpha_spending_fn='uniform',
				 boundary_strategy='random',
				 use_learner_weights=False,
				 skip_const_predictor=False,
				 tiebreak=False,
				 quit_on_rejection=True,
				 random_seed=None):
		# Save arguments
		self.protocol = protocol
		self.test_thresh = test_thresh
		self.learner = learner
		self.n_burn_in = n_burn_in
		self.reveal_batch_size = reveal_batch_size
		self.refit_batch_size = refit_batch_size
		self.n_min = n_min
		self.alpha_min = alpha_min
		self.alpha_spending_fn = alpha_spending_fn
		self.boundary_strategy = boundary_strategy
		self.use_learner_weights = use_learner_weights
		self.skip_const_predictor = skip_const_predictor
		self.tiebreak = tiebreak
		self.quit_on_rejection = quit_on_rejection
		self.random_seed = random_seed
		# Extract information
		self.n = self.protocol.Y.shape[0]
		self.alpha = self.protocol.alpha
		# Set defaults
		if 0 < self.n_burn_in < 1:
			self.n_burn_in = max(1, int(self.n * self.n_burn_in))
		if isinstance(self.n_min, str) and self.n_min == 'auto':
			self.n_min = int(max(30, 0.05 * self.n))
		if isinstance(self.alpha_min, str) and self.alpha_min == 'auto':
			self.alpha_min = 1 - np.power(1 - self.alpha, 1/20)
		if isinstance(self.alpha_spending_fn, str) and self.alpha_spending_fn == 'obf':
			z = scipy.stats.norm.ppf(1 - self.alpha / 2)
			self.alpha_spending_fn = lambda sample_eff: self.alpha - (2 - 2 * scipy.stats.norm.cdf(z / np.sqrt(sample_eff))) if sample_eff > 0 else self.alpha
		elif isinstance(self.alpha_spending_fn, str) and self.alpha_spending_fn == 'uniform':
			self.alpha_spending_fn = lambda sample_eff: self.alpha * (1 - sample_eff)
		elif isinstance(self.alpha_spending_fn, str) and self.alpha_spending_fn == 'instantaneous':
			self.alpha_spending_fn = lambda sample_eff: self.alpha * int(sample_eff < 1)
		elif isinstance(self.alpha_spending_fn, str) and self.alpha_spending_fn == 'half_uniform':
			self.alpha_spending_fn = lambda sample_eff: self.alpha * min(1, 2 * (1 - sample_eff))
		if not self.quit_on_rejection:
			protocol.toggle_warn_continuation(False)
		self.rng = np.random.RandomState(self.random_seed)
		# Tracking metrics throughout strategy
		self.curr_strategy_phase = "main"
		self.first_batch = True
		self.curr_n_pts_in_region = None
		self.curr_sample_efficiency = 1.
		self.sample_size_at_last_refit = None
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

	def _spend_up_to_alpha_budget(self):
		alpha_budget = self.alpha_spending_fn(self.curr_sample_efficiency)
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
		self.sample_size_at_last_refit = self.curr_n_pts_in_region

	def _should_refit(self):
		if self.sample_size_at_last_refit is None:
			return True
		else:
			return (self.sample_size_at_last_refit - self.curr_n_pts_in_region) >= self.refit_batch_size

	def _adaptively_shrink_to_boundary(self):
		below_boundary = True
		total_n_shrink_revealed = 0
		while below_boundary:
			# Make sure we can shrink without going below n_min
			if self.curr_n_pts_in_region <= self.n_min:
				break
			# Make sure batch size would not take us below n_min
			clipped_batch_size = min(self.curr_n_pts_in_region - self.n_min, self.reveal_batch_size)
			# Fit predictor
			if self._should_refit():
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
			total_n_shrink_revealed += n_revealed
		return total_n_shrink_revealed, below_boundary

	def _reveal_at_boundary(self, batch_size):
		# Reveal batch
		if batch_size > 0:
			if self.boundary_strategy == "random" or self.first_batch:
				# Random batch
				self.protocol.reveal_random_points(k_pts=batch_size)
			elif self.boundary_strategy == "margin":
				# Margin batch
				# Refit predictor
				if self._should_refit():
					self._fit_predictor()
				# Shrink with no limiter
				self.protocol.shrink_region(self.predictor,
											k_pts=batch_size,
											tiebreak=self.tiebreak)
			else:
				raise ValueError("Invalid argument for boundary_strategy. Must be in {random, margin}")
		# Calculate reduction in sample efficiency
		relative_sample_efficiency = (self.curr_n_pts_in_region - batch_size) / self.curr_n_pts_in_region
		# Update information
		self._update_current_information()
		return relative_sample_efficiency

	def _increment_main(self):
		if self.curr_strategy_phase != "main":
			return
		elif self.curr_n_pts_in_region < self.n_min:
			# No samples remain, we terminate
			self.curr_strategy_phase = "finished"
			return
		elif self.curr_n_pts_in_region == self.n_min:
			# If we are at or below n_min, accelerate to lowest sample efficiency and terminate after testing
			self.curr_sample_efficiency = 0
			self._spend_up_to_alpha_budget()
			self.curr_strategy_phase = "finished"
			return
		# Test up to the current alpha allocation
		rejected = self._spend_up_to_alpha_budget()
		# Quit if we reach a rejection and option is requested
		if self.quit_on_rejection and rejected:
			self.curr_strategy_phase = "finished"
			return
		# Quit if no alpha remains
		if np.isclose(self.remaining_alpha, 0):
			# No alpha remaining, we terminate
			self.curr_strategy_phase = "finished"
			return
		# Set batch size
		if self.first_batch:
			batch_size = self.n_burn_in
		else:
			batch_size = self.reveal_batch_size
		# Cache curr_sample_efficiency to add to metrics later
		curr_sample_efficiency = self.curr_sample_efficiency
		# Make sure batch size would not take us below n_min
		clipped_batch_size = min(self.curr_n_pts_in_region - self.n_min, batch_size)
		# Reveal a random batch
		relative_sample_efficiency = self._reveal_at_boundary(clipped_batch_size)
		# Adaptively shrink up to the boundary
		n_shrink_revealed, below_boundary = self._adaptively_shrink_to_boundary()
		# Update current sample efficiency
		self.curr_sample_efficiency *= relative_sample_efficiency
		# Save metrics
		curr_region_mass = self.protocol.protocol_metadata.reg_mass_est.min()
		self.metrics[self.curr_n_pts_in_region] = {"curr_sample_efficiency": curr_sample_efficiency,
												   "spent_alpha": self.spent_alpha,
												   "remaining_alpha": self.remaining_alpha,
												   "n_shrink_revealed": n_shrink_revealed,
												   "n_left_in_region": self.curr_n_pts_in_region,
												   "region_mass_estimate": curr_region_mass}
		# Toggle first batch
		if self.first_batch:
			self.first_batch = False

	def run_main(self, verbose=False):
		while self.curr_strategy_phase == "main":
			self._increment_main()
			if verbose and self.curr_strategy_phase != "finished":
				last_key = min(self.metrics.keys())
				print("METRICS =", self.metrics[last_key])
		if verbose:
			rejected = (self.protocol.get_rejected_region() is not None)
			curr_region_mass = self.protocol.protocol_metadata.reg_mass_est.min()
			print("SUMMARY: rejected = {}, spent_alpha = {}, n_left_in_region = {}, region_mass-estimate = {}".format(rejected,
																													  self.spent_alpha,
																													  self.curr_n_pts_in_region,
																													  curr_region_mass))

	def run_strategy(self, verbose=False):
		self.run_main(verbose=verbose)

