import numpy as np
import pandas as pd
import scipy.stats
import warnings

from .utils import dimension, lexicographic_gt

# ----------------------------------------- Functions ----------------------------------------- #

def post_interactive_testing_diagnostics(testing_history_df, n, min_sample_size=30, min_sample_prop=0.05,
										 max_num_reg=40, alpha_min=1e-5, sqrt_max_var_n_ratio_limit=20):
	'''
	testing_history_df: the testing_history attribute of IRST
	n: total number of samples in analysis

	This should be called after running the IRST protocol to check for potential irregularities
	'''
	# Drop the initialization row, stage_number = -1
	testing_history_df = testing_history_df.loc[testing_history_df.stage_number >= 0]
	# Initialize diagnostics
	diagnostics_df = pd.DataFrame(columns=["obs", "limit", "within_range"])
	# If dataframe is empty, return df filled with nans
	if testing_history_df.shape[0] == 0:
		diagnostics_df.loc["sample_size_absolute_limit"] = [np.nan, min_sample_size, np.nan]
		diagnostics_df.loc["sample_size_proportion_limit"] = [np.nan, min_sample_prop * n, np.nan]
		diagnostics_df.loc["number_of_tests"] = [np.nan, max_num_reg, np.nan]
		diagnostics_df.loc["minimum_alpha"] = [np.nan, alpha_min, np.nan]
		diagnostics_df.loc["sqrt_max_var_n_ratio"] = [np.nan, sqrt_max_var_n_ratio_limit, np.nan]
		return diagnostics_df
	# ...otherwise, populate diagnostics df
	diagnostics_df.loc["sample_size_absolute_limit"] = [testing_history_df.sample_size.min(),
														min_sample_size,
														testing_history_df.sample_size.min() >= min_sample_size]
	diagnostics_df.loc["sample_size_proportion_limit"] = [testing_history_df.sample_size.min(),
														  min_sample_prop * n,
														  testing_history_df.sample_size.min() >= (min_sample_prop * n)]
	diagnostics_df.loc["number_of_tests"] = [testing_history_df.shape[0],
											 max_num_reg,
											 testing_history_df.shape[0] <= max_num_reg]
	diagnostics_df.loc["minimum_alpha"] = [testing_history_df.alloc_alpha.min(),
										   alpha_min,
										   testing_history_df.alloc_alpha.min() >= alpha_min]
	# Calculate max variance x n ratio up to rejection
	rej_row = testing_history_df.loc[testing_history_df.rejection]
	if rej_row.shape[0] == 0:
		rej_stage_num = np.inf
	else:
		rej_stage_num = rej_row.iloc[0].stage_number
	sub_testing_history_df = testing_history_df.loc[testing_history_df.stage_number <= rej_stage_num]
	sub_var_n = (sub_testing_history_df.variance * sub_testing_history_df.sample_size).values
	var_n_ratios = sub_var_n / sub_var_n.reshape(-1,1)
	max_var_n_ratio = var_n_ratios.max()
	sqrt_max_var_n_ratio = np.sqrt(max_var_n_ratio)
	# Add to diagnostics
	diagnostics_df.loc["sqrt_max_var_n_ratio"] = [sqrt_max_var_n_ratio,
												  sqrt_max_var_n_ratio_limit,
												  sqrt_max_var_n_ratio <= sqrt_max_var_n_ratio_limit]
	return diagnostics_df


def trunc_binom_quant(quantile, mu, trunc_level, sample_size, rng=None):
	if quantile >= 1:
		return np.inf
	# Clip trunc level to the sample size
	trunc_level = min(trunc_level, sample_size)
	# Calculate the CDF; note for convenience we start it at -1
	z_space = np.arange(-1, trunc_level + 1)
	trunc_binom_cdf = scipy.stats.binom.cdf(z_space, sample_size, mu)
	trunc_binom_cdf /= trunc_binom_cdf[-1]
	# Calculate the lower and upper quantiles and associated probabilities
	z_upper_ind = np.where(trunc_binom_cdf > quantile)[0].min()
	z_lower_ind = np.where(trunc_binom_cdf <= quantile)[0].max()
	z_upper = z_space[z_upper_ind]
	z_lower = z_space[z_lower_ind]
	upper_cdf = trunc_binom_cdf[z_upper_ind]
	lower_cdf = trunc_binom_cdf[z_lower_ind]
	# Calculate probability of sampling z_upper
	z_upper_prob = (quantile - lower_cdf) / (upper_cdf - lower_cdf)
	# Sample quantile
	if rng is None:
		sel = np.random.binomial(1, z_upper_prob)
	else:
		sel = rng.binomial(1, z_upper_prob)
	z = z_upper * sel + z_lower * (1 - sel)
	return z

# ----------------------------------------- Classes ----------------------------------------- #

class RegisteredX:
	def __init__(self, X, tiebreak_u=np.nan, orthog_u=np.nan):
		assert dimension(X) == 2, "X must be a 2D array"
		self.X = X
		self.tiebreak_u = tiebreak_u
		self.orthog_u = orthog_u

	def subset(self, inds):
		'''
		Returns copy subsetted to inds
		'''
		regX = RegisteredX(self.X[inds], self.tiebreak_u[inds], self.orthog_u[inds])
		return regX

	def concat(self, regX):
		'''
		Returns copy of RegisteredX with current data appended atop regX
		'''
		new_regX = RegisteredX(np.vstack([self.X, regX.X]),
							   np.hstack([self.tiebreak_u, regX.tiebreak_u]),
							   np.hstack([self.orthog_u, regX.orthog_u]))
		return new_regX
		

class UnitRegistrar:
	def __init__(self, random_seed=None):
		self.rng = np.random.RandomState(random_seed)

	def register_units(self, X, random_seed=None):
		assert dimension(X) == 2, "X must be a 2D array"
		# Get random number generator
		if random_seed is not None:
			rng = np.random.RandomState(random_seed)
		else:
			rng = self.rng
		tiebreak_u = rng.uniform(0, 1, size=X.shape[0])
		orthog_u = rng.uniform(0, 1, size=X.shape[0])
		regX = RegisteredX(X, tiebreak_u, orthog_u)
		return regX


class PrioritizationScore:
	def __init__(self, predictor, tiebreak=False):
		'''
		predictor: function accepting a 2D array and returning a 1D or 2D vector of scores (if 2D, scores are ordered lexicographically)
		tiebreak: boolean indicating whether we should include a tiebreaker dimension
		'''
		self.predictor = predictor
		self.tiebreak = tiebreak

	def toggle_tiebreak(self, tiebreak):
		'''
		Toggle the tiebreak attribute
		'''
		self.tiebreak = tiebreak

	def score(self, regX):
		'''
		regX: a RegisteredX object

		return: a 2D array of scores for each point in X. If self.tiebreak = True, tiebreaker dimension is included as last column
		'''
		# First extract X from regX
		X = regX.X
		# Extract prediction from model
		scores = self.predictor(X)
		# If scores are 1D array, reshape to n x 1
		if dimension(scores) == 1:
			scores = scores.reshape(-1,1)
		# Add tiebreaker dimension, if necessary
		if self.tiebreak:
			scores = np.hstack((scores, regX.tiebreak_u.reshape(-1,1)))
		return scores


class Region:
	def __init__(self):
		# Each region constraint will consist of a PrioritizationScore object, a threshold, and optional metadata
		self.region_constraints = {}

	def add_region_constraint(self, name, pscore, thresh, metadata={}):
		# Note that if pscore returns k-dimensional scores, thresh must be a k-vector
		self.region_constraints[name] = [pscore, thresh, metadata]

	def in_region(self, regX):
		in_reg = np.ones(regX.X.shape[0], dtype=bool)
		for name in self.region_constraints.keys():
			pscore, thresh, _ = self.region_constraints[name]
			scores = pscore.score(regX)
			exceed_thresh = lexicographic_gt(scores, thresh)
			in_reg = in_reg & exceed_thresh
		return in_reg

	def subset_constraints(self, constr_fn):
		'''
		constr_fn: a function that takes in a metadata dictionary and returns True or False

		Returns an instance of Region that contains all constraints where constr_fn evaluates to True
		'''
		new_region = Region()
		for constr_name in self.region_constraints.keys():
			pscore, thresh, metadata = self.region_constraints[constr_name]
			if constr_fn(metadata):
				new_region.add_region_constraint(constr_name, pscore, thresh, metadata=metadata)
		return new_region

	def snapshot(self):
		'''
		Returns a copy of current region
		'''
		return self.subset_constraints(constr_fn=lambda x: True)


class IRST:
	def __init__(self, regX, Y, pY=None, alpha=None, alert_rejections=True):
		'''
		Interactive Region Selector and Tester (IRST)

		regX: a RegisteredX object, and its attribute X must be a 2D array
		Y: 1D array of outcomes that will be revealed to the analyst
		pY: 1D array of outcomes that will be used for testing. These may be the same as Y
		alpha: target error rate
		alert_rejections: if True, report after each test whether or not the test rejected yet,
						  and raise warnings if analyst continues past first rejection
		'''
		self.regX = regX
		self.Y = Y
		if pY is None:
			self.pY = Y
		else:
			self.pY = pY
		self.region = Region()
		self.current_stage_number = 0
		self.is_revealed = np.zeros(self.regX.X.shape[0], dtype=bool)
		# We will always suppose that regX_revealed and regX_masked have the same ordering as regX after subsetting to is_revealed, etc.
		self.regX_revealed = RegisteredX(self.regX.X[:0], self.regX.tiebreak_u[:0], self.regX.orthog_u[:0])
		self.regX_masked = RegisteredX(self.regX.X, self.regX.tiebreak_u, self.regX.orthog_u)
		self.Y_revealed = self.Y[:0]
		self.Y_masked = self.Y
		self.pY_revealed = self.pY[:0]
		self.pY_masked = self.pY
		# Protocol metadata; pt_order will also break ties by original ordering
		self.protocol_metadata = pd.DataFrame(columns=["orig_ind", "pt_order", "pseudo_pt_order", "rev_stage",
													   "is_random", "reg_mass_est", "unmasking_score",
													   "pseudo_reg_mass_est", "pseudo_unmasking_score"])
		self.protocol_metadata = self.protocol_metadata.astype({"orig_ind": int, "pt_order": int, "pseudo_pt_order": float, "rev_stage": int,
																"is_random": bool, "reg_mass_est": float, "unmasking_score": object,
																"pseudo_reg_mass_est": float, "pseudo_unmasking_score": object})
		# Testing variables
		self.alpha = alpha
		self.rejection_stage_number = None
		self.rejected_region = None
		self.alert_rejections = alert_rejections
		self.warn_continuation = True
		if self.alpha is not None:
			# Initialize testing history
			self._initialize_testing_history()
			# Update observable testing history
			self._update_observable_testing_history()

	def toggle_warn_continuation(self, warn_continuation):
		self.warn_continuation = warn_continuation

	def _initialize_testing_history(self):
		'''
		We always initialize with one row, corresponding to statistics of the full sample (but no alpha allocated, so crit_val = inf)
		This allows other methods to access things like sample size and remaining_alpha before anything has been tested
		This will correspond to a stage number of -1
		'''
		mean = self.pY_masked.mean()
		variance = self.pY_masked.var()
		sample_size = self.pY_masked.shape[0]
		if variance == 0:
			test_stat = np.nan
		else:
			test_stat = (np.sqrt(sample_size) * mean) / np.sqrt(variance)
		init_row = [-1, mean, variance, sample_size, test_stat, 0, self.alpha, np.inf, False]
		self.testing_history = pd.DataFrame([init_row], columns=["stage_number", "mean", "variance", "sample_size", "test_stat",
																 "alloc_alpha", "remaining_alpha", "crit_val", "rejection"])
		self.testing_history = self.testing_history.astype({"stage_number": int, "mean": float, "variance": float, "sample_size": int,
															"test_stat": float, "alloc_alpha": float, "remaining_alpha": float,
															"crit_val": float, "rejection": bool})

	def _check_nonzero_sample_size(self):
		if self.Y_masked.shape[0] == 0:
			raise RuntimeError("There are no masked samples remaining.")

	def _warn_continuation(self):
		'''
		Should be called whenever an action is taken. Will warn if alert_rejections = True, warn_continuation = True,
		and we have already made a rejection
		'''
		if self.warn_continuation and self.alert_rejections and (self.rejection_stage_number is not None):
			warnings.warn(("Protocol reveals information about rejections and a rejection has been made. "
						   "Further data exploration may be subject to selection bias and should be conducted cautiously."),
						  stacklevel=2)

	def _update_sub_meta_df(self, masked_scores, rev_indic, is_random):
		'''
		This should be called before _update_is_revealed and _update_masking, since it assumes is_revealed
		does NOT reflect the information in rev_indic
		'''
		# Get data from last revealed point
		if self.protocol_metadata.shape[0] > 0:
			lastrow = self.protocol_metadata.loc[self.protocol_metadata.pt_order.idxmax()]
			last_reg_mass_est = lastrow.reg_mass_est
			last_pt_order = lastrow.pt_order
		else:
			last_reg_mass_est = 1.
			last_pt_order = -1
		# Calculate proportion estimates; these assume no tiebreaks (tiebreak mass is always just min mass among tiebreaks)
		if is_random:
			prop_ests = [last_reg_mass_est] * rev_indic.sum()
		else:
			prop_reductions = (rev_indic.shape[0] - np.arange(1, 1 + rev_indic.sum())) / rev_indic.shape[0]
			prop_ests = last_reg_mass_est * prop_reductions
		orig_inds = np.where(~self.is_revealed)[0][rev_indic]
		sort_inds = np.lexsort(masked_scores[rev_indic].T[::-1])
		sorted_orig_inds = orig_inds[sort_inds]
		sorted_masked_scores = masked_scores[rev_indic][sort_inds]
		sorted_pt_order = np.arange(last_pt_order + 1, last_pt_order + 1 + len(sorted_orig_inds))
		sorted_pseudo_pt_order = [np.nan] * len(sorted_orig_inds)
		df_data = [sorted_orig_inds,
				   np.arange(last_pt_order + 1, last_pt_order + 1 + len(sorted_orig_inds)),
				   sorted_pseudo_pt_order,
				   [self.current_stage_number] * len(sorted_orig_inds),
				   [is_random] * len(sorted_orig_inds),
				   prop_ests]
		sub_meta_df = pd.DataFrame(np.array(df_data).T, columns=["orig_ind", "pt_order", "pseudo_pt_order",
																 "rev_stage", "is_random", "reg_mass_est"])
		sub_meta_df["unmasking_score"] = pd.Series([tuple(s) for s in sorted_masked_scores]).values
		sub_meta_df["pseudo_reg_mass_est"] = np.nan
		sub_meta_df["pseudo_unmasking_score"] = np.nan
		sub_meta_df = sub_meta_df.astype({"orig_ind": int, "pt_order": int, "pseudo_pt_order": float, "rev_stage": int,
										  "is_random": bool, "reg_mass_est": float, "unmasking_score": object,
										  "pseudo_reg_mass_est": float, "pseudo_unmasking_score": object})
		# Concat with global metadata and reorder
		if self.protocol_metadata.shape[0] > 0:
			self.protocol_metadata = (pd.concat([self.protocol_metadata, sub_meta_df], axis=0, ignore_index=True)
									  .sort_values(by="orig_ind", ignore_index=True))
		else:
			self.protocol_metadata = sub_meta_df.sort_values(by="orig_ind", ignore_index=True)

	def _update_pseudo_pt_orders(self, masked_scores, rev_indic, masked_orig_inds,
								 reorderable_scores, reorderable_orig_inds, total_num_reorderables):
		'''
		This should be called before _update_is_revealed and _update_masking, but (immediately) after _update_sub_meta_df
		Note that ties among masked_scores and reorderable_scores are broken according to the order of the original points
			WITHIN each group (masked vs. reorderable). Thus, a fully tied collection would rank all masked_scores before
			all reorderable_scores, but otherwise respect the order of the original rows
		'''
		# Get data from last pseudo-ordered point
		if self.protocol_metadata.pseudo_pt_order.isnull().all():
			last_pseudo_reg_mass_est = 1.
			last_pseudo_pt_order = -1
		else:
			lastrow = self.protocol_metadata.loc[self.protocol_metadata.pseudo_pt_order.idxmax()]
			last_pseudo_reg_mass_est = lastrow.pseudo_reg_mass_est
			last_pseudo_pt_order = lastrow.pseudo_pt_order
		# Calculate proportion estimates; these assume no tiebreaks (tiebreak mass is always just min mass among tiebreaks)
		pseudo_total = rev_indic.shape[0] + total_num_reorderables
		pseudo_num_revealed = rev_indic.sum() + len(reorderable_orig_inds)
		prop_reductions = (pseudo_total - np.arange(1, 1 + pseudo_num_revealed)) / pseudo_total
		prop_ests = last_pseudo_reg_mass_est * prop_reductions
		# Sort scores and indices
		all_orig_inds = np.hstack((masked_orig_inds, reorderable_orig_inds))
		all_scores = np.vstack((masked_scores[rev_indic], reorderable_scores))
		sort_inds = np.lexsort(all_scores.T[::-1])
		sorted_orig_inds = all_orig_inds[sort_inds]
		sorted_scores = all_scores[sort_inds]
		sorted_pseudo_pt_order = np.arange(last_pseudo_pt_order + 1, last_pseudo_pt_order + 1 + len(sorted_orig_inds))
		# Update protocol_metadata
		ord_meta_df_indices = (self.protocol_metadata
							   .loc[:,["orig_ind"]].reset_index(names="index")
							   .set_index("orig_ind")
							   .loc[sorted_orig_inds]
							   .loc[:,"index"].values)
		self.protocol_metadata.loc[ord_meta_df_indices, "pseudo_pt_order"] = sorted_pseudo_pt_order
		self.protocol_metadata.loc[ord_meta_df_indices, "pseudo_reg_mass_est"] = prop_ests
		self.protocol_metadata.loc[ord_meta_df_indices, "pseudo_unmasking_score"] = pd.Series([tuple(s) for s in sorted_scores]).values
		
	def _update_is_revealed(self, new_revealed_indic):
		'''
		new_revealed_indic: boolean array of length [# of masked points]
							True indicates the point is now revealed, False indicates still masked

		We update self.is_revealed based on this, supposing that points in self.regX_masked are currently subsetted
		to self.is_revealed respecting the original ordering (and similarly for Y, pY)
		'''
		global_masked_inds = np.where(~self.is_revealed)[0]
		new_revealed_inds = global_masked_inds[new_revealed_indic]
		self.is_revealed[new_revealed_inds] = True

	def _update_masking(self):
		'''
		Updates self.regX, self.Y, and self.pY (_revealed and _masked) to reflect the current state of self.is_revealed
		'''
		self.regX_revealed = self.regX.subset(self.is_revealed)
		self.regX_masked = self.regX.subset(~self.is_revealed)
		self.Y_revealed = self.Y[self.is_revealed]
		self.Y_masked = self.Y[~self.is_revealed]
		self.pY_revealed = self.pY[self.is_revealed]
		self.pY_masked = self.pY[~self.is_revealed]

	def _calculate_truncation_level(self, curr_variance, curr_sample_size):
		'''
		Calculate current truncation level of test statistic based on information in Y_revealed, protocol_metadata, testing_history
		'''
		# If current or any previously encountered variance is 0, trunc_level = np.nan
		if (curr_variance == 0) | pd.isnull(self.testing_history.variance).any():
			trunc_level = np.nan
			return np.nan
		# ...otherwise, we calculate using the usual formula
		all_trunc_s = []
		for s, row in self.testing_history.iterrows():
			cv_s = row.crit_val
			# Get the relevant indices (points that were revealed between row.stage_number and current)
			relevant_indic = (self.protocol_metadata.rev_stage > row.stage_number)
			# Calculate intermediate quantities
			delta_st = self.pY_revealed[relevant_indic].sum() / np.sqrt(row.variance * row.sample_size)
			v_st = np.sqrt((curr_variance * curr_sample_size) / (row.variance * row.sample_size))
			# Calculate the sub-truncation
			trunc_s = (cv_s - delta_st) / v_st
			all_trunc_s.append(trunc_s)
		# Total truncation is minimum of sub-truncations
		if len(all_trunc_s) == 0:
			trunc_level = np.inf
		else:
			trunc_level = np.min(all_trunc_s)
		return trunc_level

	def _calculate_critical_value(self, quantile, trunc_level, lower_clip=0):
		cv = scipy.stats.norm.ppf(quantile * scipy.stats.norm.cdf(trunc_level))
		clipped_cv = np.clip(cv, a_min=lower_clip, a_max=None)
		return clipped_cv

	def _calculate_new_error_budget(self, budget, allocation):
		new_beta = (1 - budget) / (1 - allocation)
		return 1 - new_beta

	def _update_observable_testing_history(self):
		if self.alert_rejections:
			self.observable_testing_history = self.testing_history.loc[:,["stage_number", "sample_size", "alloc_alpha",
																		  "remaining_alpha", "rejection"]]
		else:
			self.observable_testing_history = self.testing_history.loc[:,["stage_number", "sample_size", "alloc_alpha", "remaining_alpha"]]

	def shrink_region(self, predictor, tiebreak=False, max_thresh=None, k_pts=1, reorderable_scores=None):
		'''
		Reveals all points whose scores are no greater than the k_pts smallest priority score IF that score is <= max_thresh
		Otherwise, shrink to region defined by pscore > max_thresh

		Notes:
			- We may reveal more that k_pts if there are ties and tiebreaking is not enabled
			- If tiebreak = True but max_thresh is short one dimension, we implicitly interpret it as [max_thresh, 1]
			- If reorderable_scores is provided, it should not contain the tiebreaker dimension. That will be automatically added as necessary
		'''
		# Check that there are still samples remaining
		self._check_nonzero_sample_size()
		# Warn if continued exploration may be biased
		self._warn_continuation()
		# First build PrioritizationScore object
		pscore = PrioritizationScore(predictor, tiebreak)
		# Score masked points
		masked_scores = pscore.score(self.regX_masked)
		pscore_dim = masked_scores.shape[1]
		# Process max_thresh if necessary
		if max_thresh is not None:
			if len(max_thresh) < pscore_dim:
				max_thresh = np.hstack((max_thresh, [1]))
		else:
			max_thresh = np.empty(pscore_dim, dtype=float)
			max_thresh[:] = np.inf
		# Get kth smallest priority score (lexicographic)
		k_pts = min(k_pts, masked_scores.shape[0])
		thresh = masked_scores[np.lexsort(masked_scores.T[::-1])[k_pts - 1]]
		# Cap at max_thresh
		if lexicographic_gt(np.array([thresh]), max_thresh)[0]:
			thresh = max_thresh
		# Get indicators of revealed points among regX_masked
		masked_indic = lexicographic_gt(masked_scores, thresh)
		rev_indic = ~masked_indic
		masked_orig_inds = np.where(~self.is_revealed)[0][rev_indic]
		# Extract random samples still in region
		total_num_reorderables = self.protocol_metadata.pseudo_pt_order.isnull().sum()
		reorderable_indic = self.protocol_metadata.pseudo_pt_order.isnull().values
		reorderable_orig_inds = self.protocol_metadata.loc[reorderable_indic].orig_ind.values
		if reorderable_scores is None:
			# If custom scores are not provided, score based on the given prioritization function
			if total_num_reorderables > 0:
				reorderable_scores = pscore.score(self.regX_revealed.subset(reorderable_indic))
			else:
				reorderable_scores = np.zeros((0, pscore_dim))
		elif tiebreak:
			# Add tiebreaker dimension if necessary
			reorderable_scores = np.hstack((reorderable_scores, regX_revealed.tiebreak_u[reorderable_indic].reshape(-1,1)))
		# Only those with scores at or below threshold can be reordered
		sub_reorderable_indic = ~lexicographic_gt(reorderable_scores, thresh)
		reorderable_scores = reorderable_scores[sub_reorderable_indic]
		reorderable_orig_inds = reorderable_orig_inds[sub_reorderable_indic]
		# Update subset protocol metadata information
		self._update_sub_meta_df(masked_scores, rev_indic, is_random=False)
		# Update pseudo-orders for random samples still in region
		self._update_pseudo_pt_orders(masked_scores, rev_indic, masked_orig_inds,
									  reorderable_scores, reorderable_orig_inds, total_num_reorderables)
		# Update region
		metadata = {"stage_number": self.current_stage_number,
					"relative_mass_est": 1 - rev_indic.mean(),
					"mass_est": self.protocol_metadata.reg_mass_est.min()}
		self.region.add_region_constraint("stage_{}".format(self.current_stage_number), pscore, thresh, metadata=metadata)
		# Extract revealed points to show the analyst immediately
		rev_X = self.regX_masked.X[rev_indic]
		rev_Y = self.Y_masked[rev_indic]
		# Update the global revealed indices (need to match indexing)
		self._update_is_revealed(rev_indic)
		# Update the revealed/masked data structures
		self._update_masking()
		# Update stage number
		self.current_stage_number += 1
		# Return most recently revealed information
		return rev_X, rev_Y, thresh

	def reveal_random_points(self, k_pts=1):
		# Check that there are still samples remaining
		self._check_nonzero_sample_size()
		# Warn if continued exploration may be biased
		self._warn_continuation()
		# Reveal points without modifying the region. We use the orthog_u, the orthogonal dimension
		rev_inds = np.argsort(self.regX_masked.orthog_u)[:k_pts]
		rev_X = self.regX_masked.X[rev_inds]
		rev_Y = self.Y[rev_inds]
		# Calculate indicator vector of revealed indices
		rev_indic = np.zeros(self.regX_masked.X.shape[0], dtype=bool)
		rev_indic[rev_inds] = True
		# Create subset protocol metadata information
		self._update_sub_meta_df(self.regX_masked.orthog_u.reshape(-1,1), rev_indic, is_random=True)
		# Update the global revealed indices (need to match indexing)
		self._update_is_revealed(rev_indic)
		# Update the revealed/masked data structures
		self._update_masking()
		# Update stage number
		self.current_stage_number += 1
		# Return most recently revealed information
		return rev_X, rev_Y

	def allocate_alpha_and_test(self, alloc_alpha, critval_lower_clip=0):
		# Check that there are still samples remaining
		self._check_nonzero_sample_size()
		# Check that self.alpha is not None
		if self.alpha is None:
			raise ValueError("Cannot test if alpha is not defined at class instantiation.")
		# Warn if continued exploration may be biased
		self._warn_continuation()
		# Get remaining alpha
		remaining_alpha = self.testing_history.remaining_alpha.min()
		# Raise an error if analyst has allocated above what is allowed
		if (alloc_alpha > remaining_alpha) and not np.isclose(alloc_alpha - remaining_alpha, 0):
			raise ValueError("alloc_alpha was given the value {}, which is beyond the remaining error budget {}"
							 .format(alloc_alpha, remaining_alpha))
		# Calculate mean, variance, sample size, test statistic
		mean = self.pY_masked.mean()
		variance = self.pY_masked.var()
		sample_size = self.pY_masked.shape[0]
		if variance == 0:
			test_stat = np.nan
		else:
			test_stat = (np.sqrt(sample_size) * mean) / np.sqrt(variance)        
		# Check if a rejection has been encountered yet; if so, no need to calculate critical value
		if self.rejection_stage_number is not None:
			crit_val = np.nan
		else:
			# Calculate implied truncation level, given no previous rejections
			trunc_level = self._calculate_truncation_level(variance, sample_size)
			crit_val = self._calculate_critical_value(1 - alloc_alpha, trunc_level, lower_clip=critval_lower_clip)
		# Calculate new remaining alpha
		new_remaining_alpha = self._calculate_new_error_budget(remaining_alpha, alloc_alpha)
		# Check for rejection
		rejection = False
		if (self.rejection_stage_number is None) and (test_stat > crit_val):
			# Set rejection stage number to current stage
			self.rejection_stage_number = self.current_stage_number
			# Take snapshot of current region
			self.rejected_region = self.region.snapshot()
			# Set rejection variable to True; to be used below
			rejection = True
		# Update testing history
		new_row = [self.current_stage_number, mean, variance, sample_size, test_stat, alloc_alpha, new_remaining_alpha, crit_val, rejection]
		new_row_df = pd.DataFrame([new_row], columns=self.testing_history.columns)
		new_row_df = new_row_df.astype({"stage_number": int, "mean": float, "variance": float, "sample_size": int,
										"test_stat": float, "alloc_alpha": float, "remaining_alpha": float,
										"crit_val": float, "rejection": bool})
		self.testing_history = pd.concat([self.testing_history, new_row_df], axis=0, ignore_index=True)
		# Update observable testing history
		self._update_observable_testing_history()
		# Increment stage
		self.current_stage_number += 1

	def get_current_information(self, cov_only=True):
		if cov_only:
			return self.regX_revealed.X, self.Y_revealed, self.protocol_metadata
		else:
			return self.regX_revealed, self.Y_revealed, self.protocol_metadata

	def get_reorderable_indices(self):
		reorderable_indic = self.protocol_metadata.pseudo_pt_order.isnull().values
		return reorderable_indic

	def get_num_remaining_samples(self):
		return self.Y_masked.shape[0]

	def get_observable_testing_history(self):
		return self.observable_testing_history

	def get_current_region(self):
		return self.region

	def get_rejected_region(self):
		return self.rejected_region


class IRSTBinary(IRST):
	def __init__(self, regX, Y, test_thresh, alpha=None, alert_rejections=True, random_seed=None):
		'''
		Interactive Region Selector and Tester (IRST) for binary outcomes
		'''
		super().__init__(regX=regX, Y=Y, alpha=alpha, alert_rejections=alert_rejections)
		self.test_thresh = test_thresh
		self.random_seed = random_seed
		self.rng = np.random.RandomState(self.random_seed)

	def _initialize_testing_history(self):
		'''
		We always initialize with one row, corresponding to statistics of the full sample (but no alpha allocated, so crit_val = inf)
		This allows other methods to access things like sample size and remaining_alpha before anything has been tested
		This will correspond to a stage number of -1
		'''
		test_stat = self.Y_masked.sum()
		sample_size = self.Y_masked.shape[0]
		init_row = [-1, test_stat, sample_size, 0, self.alpha, np.inf, False]
		self.testing_history = pd.DataFrame([init_row], columns=["stage_number", "test_stat", "sample_size",
																 "alloc_alpha", "remaining_alpha", "crit_val", "rejection"])
		self.testing_history = self.testing_history.astype({"stage_number": int, "test_stat": int, "sample_size": int,
															"alloc_alpha": float, "remaining_alpha": float,
															"crit_val": float, "rejection": bool})

	def _calculate_truncation_level(self):
		'''
		Calculate current truncation level of test statistic based on information in Y_revealed, protocol_metadata, testing_history
		'''
		all_trunc_s = []
		for s, row in self.testing_history.iterrows():
			cv_s = row.crit_val
			# Get the relevant indices (points that were revealed between row.stage_number and current)
			relevant_indic = (self.protocol_metadata.rev_stage > row.stage_number)
			# Calculate intermediate quantity
			delta_st = self.pY_revealed[relevant_indic].sum()
			# Calculate the sub-truncation
			trunc_s = cv_s - delta_st
			all_trunc_s.append(trunc_s)
		# Total truncation is minimum of sub-truncations
		if len(all_trunc_s) == 0:
			trunc_level = np.inf
		else:
			trunc_level = np.min(all_trunc_s)
		return trunc_level

	def _calculate_critical_value(self, quantile, trunc_level, sample_size):
		cv = trunc_binom_quant(quantile, self.test_thresh, trunc_level, sample_size, self.rng)
		return cv
		
	def allocate_alpha_and_test(self, alloc_alpha):
		# Check that there are still samples remaining
		self._check_nonzero_sample_size()
		# Check that self.alpha is not None
		if self.alpha is None:
			raise ValueError("Cannot test if alpha is not defined at class instantiation.")
		# Warn if continued exploration may be biased
		self._warn_continuation()
		# Get remaining alpha
		remaining_alpha = self.testing_history.remaining_alpha.min()
		# Raise an error if analyst has allocated above what is allowed
		if (alloc_alpha > remaining_alpha) and not np.isclose(alloc_alpha - remaining_alpha, 0):
			raise ValueError("alloc_alpha was given the value {}, which is beyond the remaining error budget {}"
							 .format(alloc_alpha, remaining_alpha))
		# Calculate test statistic and sample size
		test_stat = self.Y_masked.sum()
		sample_size = self.Y_masked.shape[0]   
		# Check if a rejection has been encountered yet; if so, no need to calculate critical value
		if self.rejection_stage_number is not None:
			crit_val = np.nan
		else:
			# Calculate implied truncation level, given no previous rejections
			trunc_level = self._calculate_truncation_level()
			crit_val = self._calculate_critical_value(1 - alloc_alpha, trunc_level, sample_size)
		# Calculate new remaining alpha
		new_remaining_alpha = self._calculate_new_error_budget(remaining_alpha, alloc_alpha)
		# Check for rejection
		rejection = False
		if (self.rejection_stage_number is None) and (test_stat > crit_val):
			# Set rejection stage number to current stage
			self.rejection_stage_number = self.current_stage_number
			# Take snapshot of current region
			self.rejected_region = self.region.snapshot()
			# Set rejection variable to True; to be used below
			rejection = True
		# Update testing history
		new_row = [self.current_stage_number, test_stat, sample_size, alloc_alpha, new_remaining_alpha, crit_val, rejection]
		new_row_df = pd.DataFrame([new_row], columns=self.testing_history.columns)
		new_row_df = new_row_df.astype({"stage_number": int, "test_stat": int, "sample_size": int,
										"alloc_alpha": float, "remaining_alpha": float,
										"crit_val": float, "rejection": bool})
		self.testing_history = pd.concat([self.testing_history, new_row_df], axis=0, ignore_index=True)
		# Update observable testing history
		self._update_observable_testing_history()
		# Increment stage
		self.current_stage_number += 1
