import numpy as np
import scipy.stats

from ...protocol.IRST import IRST, IRSTBinary, UnitRegistrar

# ---------------------------------------- Helper functions ---------------------------------------- #

def wmean_se(x, w=None):
	"""
	Weighted mean + standard error for 'usual' (frequency/probability) weights.
	Scaling w by any constant does not change the result.
	Returns (mu, se) where se ≈ sqrt(s_w^2 / n_eff).
	"""
	x = np.asarray(x, float)
	w = np.ones_like(x) if w is None else np.asarray(w, float)

	S = w.sum()
	S2 = np.sum(w * w)

	mu = np.sum(w * x) / S

	# Unbiased weighted variance:
	denom = S - S2 / S                 # = S * (1 - sum(a_i^2)), with a_i = w_i / S
	if denom <= 0:                      # degenerate (e.g., one nonzero weight)
		return mu, np.nan

	s2 = np.sum(w * (x - mu)**2) / denom
	se = np.sqrt(s2 * S2 / S**2)        # = sqrt(s2 / n_eff), n_eff = S**2 / S2

	return mu, se

# ---------------------------------------- Original chiseling wrapper with modifications ---------------------------------------- #

class ChiselingInterpretable:
	'''
	This is a wrapper that instantiates protocol and runs SimpleMarginStrategyInterpretable
	Note that WLOG non-binary chiseling tests mean <= 0. test_thresh only affects behavior of strategy class and binary chiseling.
	Thus, if binary = False, should either have test_thresh = 0 or pY should be provided and reflect subtracting by test_thresh
	'''
	def __init__(self,
				 X,
				 Y,
				 test_thresh,
				 alpha,
				 learner,
				 n_burn_in,
				 pY=None,
				 binary=False,
				 alpha_init=0,
				 refit_batch_prop=0.05,
				 reveal_batch_prop=0.01,
				 margin_width=1,
				 n_min='auto',
				 alpha_min='auto',
				 use_learner_weights=False,
				 skip_const_predictor=False,
				 shrink_to_boundary=True,
				 tiebreak=False,
				 min_box_cond_samps=30,
				 ignored_facets=[],
				 discrete_coords=[],
				 random_seed=None):
		self.X = X
		self.Y = Y
		self.test_thresh = test_thresh
		self.alpha = alpha
		self.learner = learner
		self.n_burn_in = n_burn_in
		self.pY = pY
		self.binary = binary
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
		self.min_box_cond_samps = min_box_cond_samps
		self.ignored_facets = ignored_facets
		self.discrete_coords = discrete_coords
		self.rng = np.random.RandomState(random_seed)
		# Register units
		rs = self.rng.randint(0, 2**32 - 1)
		self.unit_reg = UnitRegistrar(random_seed=rs)
		self.regX = self.unit_reg.register_units(X)
		# Initialize protocol
		if binary:
			rs = self.rng.randint(0, 2**32 - 1)
			self.protocol = IRSTBinary(regX=self.regX, Y=self.Y, test_thresh=self.test_thresh, alpha=self.alpha, random_seed=rs)
		else:
			self.protocol = IRST(regX=self.regX, Y=self.Y, pY=self.pY, alpha=self.alpha)
		# Initialize strategy
		self.strategy = SimpleMarginStrategyInterpretable(protocol=self.protocol,
														  test_thresh=self.test_thresh,
														  learner=self.learner,
														  n_burn_in=self.n_burn_in,
														  alpha_init=self.alpha_init,
														  refit_batch_prop=self.refit_batch_prop,
														  reveal_batch_prop=self.reveal_batch_prop,
														  margin_width=self.margin_width,
														  n_min=self.n_min,
														  alpha_min=self.alpha_min,
														  use_learner_weights=self.use_learner_weights,
														  skip_const_predictor=self.skip_const_predictor,
														  shrink_to_boundary=self.shrink_to_boundary,
														  tiebreak=self.tiebreak,
														  min_box_cond_samps=self.min_box_cond_samps,
														  ignored_facets=self.ignored_facets,
														  discrete_coords=self.discrete_coords)

	def run_strategy(self, verbose=False):
		self.strategy.run_strategy(verbose=verbose)


# ---------------------------------------- Original strategy class with modifications ---------------------------------------- #

class SimpleMarginStrategyInterpretable:
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
				 tiebreak=False,
				 min_box_cond_samps=30,
				 ignored_facets=[],
				 discrete_coords=[]):
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
		self.min_box_cond_samps = min_box_cond_samps
		self.ignored_facets = ignored_facets
		self.discrete_coords = discrete_coords
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
		self.reveal_batch_size = max(1, int(self.n * self.reveal_batch_prop))
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
		# Facet variables
		self.curr_min_facet = None
		self.curr_min_facet_sign = None
		self.curr_min_facet_value = -np.inf
		self.curr_min_facet_f = None
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

	def _should_refit(self):
		if self.sample_size_at_last_refit is None:
			return True
		else:
			return (self.sample_size_at_last_refit - self.curr_n_pts_in_region) >= self.refit_batch_size

	def _estimate_facet_value(self, coord, facet_boundary, box_cond_X, sample_weight):
		# Impute facet_boundary at the coord'th coordinate of box_cond_X
		boundary_X = np.array(box_cond_X)
		boundary_X[:,coord] = facet_boundary
		# Evaluate the predictor at these values
		boundary_preds = self.predictor(boundary_X).flatten()
		# Calculate the weighted mean
		facet_val_est, facet_val_se = wmean_se(boundary_preds, w=sample_weight)
		return facet_val_est, facet_val_se

	def _update_facet_state(self):
		d = self.revX.shape[1]
		# Calculate in-region indices
		revX_registered, _, _ = self.protocol.get_current_information(cov_only=False)
		in_reg = self.protocol.region.in_region(revX_registered)
		# Also calculate the sample weight
		sample_weight = np.ones(self.revX.shape[0])
		if in_reg.any():
			sample_weight[in_reg] = (self.n - self.revX.shape[0] + in_reg.sum()) / in_reg.sum()
		# Extract the boundary values for each facet. We approximate this by taking the coordinate-wise range of points in the region.
		facet_lower, facet_upper = self.revX[in_reg].min(axis=0), self.revX[in_reg].max(axis=0)
		facet_lower_values, facet_upper_values = np.inf * np.ones(d), np.inf * np.ones(d)
		facet_lower_ses, facet_upper_ses = np.zeros(d), np.zeros(d)
		# Calculate coordinate-wise "distance" to region (if in region, distance is 0), standardized by coordinate standard deviations
		coord_reg_dist = np.maximum((facet_lower - self.revX) / self.revX.std(axis=0), (self.revX - facet_upper) / self.revX.std(axis=0))
		coord_reg_dist = np.maximum(coord_reg_dist, 0)
		for j in range(d):
			# Skip if we should ignore facet
			if j in self.ignored_facets:
				continue
			# Skip if X[j] is constant among in_region indices
			if np.isclose(self.revX[in_reg][:,j].var(), 0):
				continue
			# Get the relevant data that satisfies other box constraints (take all points in box, then also nearest points until we have self.min_box_cond_samps samples)
			keep_cols = np.ones(d).astype(bool)
			keep_cols[self.ignored_facets] = False
			keep_cols[j] = False
			coord_total_dists = coord_reg_dist[:,keep_cols].mean(axis=1)
			box_cond_inds = (coord_total_dists == 0)
			if box_cond_inds.sum() < self.min_box_cond_samps:
				take_inds = np.argsort(coord_total_dists)[:self.min_box_cond_samps]
				box_cond_inds[take_inds] = True
			box_cond_X = self.revX[box_cond_inds]
			# Calculate lower and upper facet values
			facet_lower_values[j], facet_lower_ses[j] = self._estimate_facet_value(j, facet_lower[j], box_cond_X, sample_weight[box_cond_inds])
			facet_upper_values[j], facet_upper_ses[j] = self._estimate_facet_value(j, facet_upper[j], box_cond_X, sample_weight[box_cond_inds])
		# Estimate the debiased minimum
		comb_facet_vals, comb_facet_ses = np.hstack((facet_lower_values, facet_upper_values)), np.hstack((facet_lower_ses, facet_upper_ses))
		debiased_min_facet_val = comb_facet_vals.min()
		# Select the best facet
		if facet_lower_values.min() <= facet_upper_values.min():
			self.curr_min_facet = np.argmin(facet_lower_values)
			self.curr_min_facet_sign = 1
			self.curr_min_facet_value = debiased_min_facet_val
			self.curr_min_facet_f = lambda x, j=self.curr_min_facet: x[:,[j]]
		else:
			self.curr_min_facet = np.argmin(facet_upper_values)
			self.curr_min_facet_sign = -1
			self.curr_min_facet_value = debiased_min_facet_val
			self.curr_min_facet_f = lambda x, j=self.curr_min_facet: -x[:,[j]]
		# Check that facet is not among ignored facets; if so, just take the first valid facet and positive direction
		if self.curr_min_facet in self.ignored_facets:
			valid_facets = [j for j in range(d) if j not in self.ignored_facets]
			self.curr_min_facet = min(valid_facets)
			self.curr_min_facet_sign = 1
			self.curr_min_facet_value = np.inf
			self.curr_min_facet_f = lambda x, j=self.curr_min_facet: x[:,[j]]

	def _fit_predictor(self):
		# Only refit the predictor if we should refit
		if self._should_refit():
			# Otherwise, refit
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
		# But always update the facet state
		self._update_facet_state()

	def _adaptively_shrink_to_boundary(self):
		while self.curr_min_facet_value < self.test_thresh:
			if self.verbose:
				print("SUMMARY (phase = shrink to boundary): n pts in region = {}, facet coord = {}, facet sign = {}, facet val = {}".format(self.curr_n_pts_in_region,
																																			 self.curr_min_facet,
																																			 self.curr_min_facet_sign,
																																			 self.curr_min_facet_value))
			# Make sure we can shrink without going below n_min
			if self.curr_n_pts_in_region <= self.n_min:
				break
			# Make sure batch size would not take us below n_min
			clipped_batch_size = min(self.curr_n_pts_in_region - self.n_min, self.reveal_batch_size)
			# Fit predictor
			self._fit_predictor()
			# Exit if predictor is constant, if setting is toggled
			if self.skip_const_predictor and "is_const" in self.predictor_meta.keys() and self.predictor_meta["is_const"]:
				break
			# Shrink region up to boundary
			if self.curr_min_facet in self.discrete_coords:
				# If shrinking along a discrete facet, we only shrink one level (suffices to let k_pts = 1 since we're not tiebreaking)
				self.protocol.shrink_region(self.curr_min_facet_f, k_pts=1)
			else:
				self.protocol.shrink_region(self.curr_min_facet_f, k_pts=clipped_batch_size)
			self._update_current_information()
		if self.verbose:
			print("SUMMARY (phase = shrink to boundary): n pts in region = {}, facet coord = {}, facet sign = {}, facet val = {}".format(self.curr_n_pts_in_region,
																																		 self.curr_min_facet,
																																		 self.curr_min_facet_sign,
																																		 self.curr_min_facet_value))

	def _shrink_region(self, batch_size):
		if batch_size > 0:
			# Refit predictor
			self._fit_predictor()
			if self.skip_const_predictor and "is_const" in self.predictor_meta.keys() and self.predictor_meta["is_const"]:
				# If predictor is constant, we just reveal points randomly
				self.protocol.reveal_random_points(k_pts=batch_size)
			else:
				# Shrink with no limiter
				if self.curr_min_facet in self.discrete_coords:
					# If shrinking along a discrete facet, we only shrink one level (suffices to let k_pts = 1 since we're not tiebreaking)
					self.protocol.shrink_region(self.curr_min_facet_f, k_pts=1)
				else:
					self.protocol.shrink_region(self.curr_min_facet_f, k_pts=batch_size)
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
												   "region_mass_estimate": curr_region_mass,
												   "curr_min_facet": self.curr_min_facet,
												   "curr_min_facet_value": self.curr_min_facet_value}

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
								 "region_mass_estimate": curr_region_mass,
								 "curr_min_facet": self.curr_min_facet,
								 "curr_min_facet_value": self.curr_min_facet_value}
		if verbose:
			print("SUMMARY=", self.metrics["FINAL"])

	def run_strategy(self, verbose=False):
		# Set a global verbosity so other methods can reference
		self.verbose=verbose
		self.run_initialize(verbose=verbose)
		self.run_main(verbose=verbose)
