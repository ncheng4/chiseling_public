import numpy as np
import pandas as pd
import scipy.stats
import inspect
import warnings

import chiseling.source.protocol.utils as utils
from ..source.protocol.IRST import IRST, IRSTBinary, UnitRegistrar
from ..source.strategies.chiseling import Chiseling
from ..source.strategies.data_splitting_strategy import DataSplittingStrategy
from ..source.strategies.simul_data_splitting_strategy import SimulDataSplittingStrategy
from ..source.strategies.bonferroni_combiner import BonferroniCombiner
from ..source.strategies.ttest_strategy import TTestStrategy
from ..source.learners.baselearners_general import linreg_learner, ridgecv_learner, make_lassocv_learner, make_elasticnetcv_learner, make_random_forest_learner, make_regularized_random_forest_learner, make_mlp_regressor_learner, make_lassocv_selector_learner, linreg_l2_learner
from ..source.learners.baselearners_binary import logreg_learner, logregl1_learner, make_logregcv_learner
from ..source.learners.baselearners_causal import causal_linreg_learner, make_causal_random_forest_learner, make_causal_random_forest_classifier_learner, make_nonneg_rct_causal_random_forest_learner

from ..dgps.basic_linear_rct import BasicLinearRCT
from ..dgps.binary_design_rct import BinaryDesignRCT
from ..dgps.kang_schafer_rct import KangSchaferRCT
from ..dgps.basic_binary_regression import BasicBinaryRegression
from ..dgps.heterogeneous_linear_rct import HeterogeneousLinearRCT
from ..dgps.nonneg_rct import NonNegRCT
from ..dgps.linear_nonneg_rct import LinearNonNegRCT
from ..dgps.benchtm import BenchTM
from ..dgps.bart_dataset import BARTDataset
from ..dgps.null_dgps import NullDGPs, null_dgps_aipw_transform

def get_function_arguments(func):
	# Get the signature of the function
	signature = inspect.signature(func)
	# Extract the parameter names as a list of strings
	return [param for param in signature.parameters]


def subset_dict(original_dict, keys_to_keep, require_all=False):
	if require_all:
		sub_d = {key: original_dict[key] for key in keys_to_keep}
	else:
		sub_d = {key: original_dict[key] for key in keys_to_keep if key in original_dict}
	return sub_d


class Benchmark:

	# ---------------------------------------------------- INITIALIZATION ---------------------------------------------------- #

	def __init__(self, settings):
		self.rng = np.random.RandomState(settings["random_seed"])
		self.settings = settings
		self._process_settings()
		self.simulation_results_data = []
		self.benchtm_cache = None

	def _process_settings(self):
		self.processed_settings = dict(self.settings)
		# Force some types
		for int_param in ["n_min", "n_burn_in"]:
			if int_param in self.processed_settings.keys() and not np.isnan(self.processed_settings[int_param]):
				if int_param == "n_burn_in" and (0 < self.processed_settings[int_param] < 1):
					pass
				else:
					self.processed_settings[int_param] = int(self.processed_settings[int_param])
		# If strategy is oracle data splitting, we double the sample size and set the strategy to 50/50 data splitting
		if self.processed_settings["strategy"] == "OracleStrategy":
			self.processed_settings["strategy"] = "DataSplittingStrategy"
			self.processed_settings["train_ratio"] = 0.5
			self.processed_settings["n"] = 2 * self.processed_settings["n"]
		# If strategy is oracle simultaneous data splitting, we double the sample size and set the strategy to 50/50 simultaneous data splitting
		if self.processed_settings["strategy"] == "OracleSimulStrategy":
			self.processed_settings["strategy"] = "SimulDataSplittingStrategy"
			self.processed_settings["train_ratio"] = 0.5
			self.processed_settings["n"] = 2 * self.processed_settings["n"]
		# Get learner from string option
		if "learner" in self.processed_settings.keys():
			if self.processed_settings["learner"] == "linreg_learner":
				self.processed_settings["learner"] = linreg_learner
			elif self.processed_settings["learner"] == "ridgecv_learner":
				self.processed_settings["learner"] = ridgecv_learner
			elif self.processed_settings["learner"] == "lassocv_learner":
				self.processed_settings["learner"] = make_lassocv_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "elasticnetcv_learner":
				self.processed_settings["learner"] = make_elasticnetcv_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "random_forest_learner":
				self.processed_settings["learner"] = make_random_forest_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "logregcv_learner":
				self.processed_settings["learner"] = make_logregcv_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "logreg_learner":
				self.processed_settings["learner"] = logreg_learner
			elif self.processed_settings["learner"] == "logregl1_learner":
				self.processed_settings["learner"] = logregl1_learner
			elif self.processed_settings["learner"] == "causal_linreg_learner":
				self.processed_settings["learner"] = causal_linreg_learner
			elif self.processed_settings["learner"] == "causal_random_forest_learner":
				self.processed_settings["learner"] = make_causal_random_forest_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "causal_random_forest_classifier_learner":
				self.processed_settings["learner"] = make_causal_random_forest_classifier_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "regularized_random_forest_learner":
				self.processed_settings["learner"] = make_regularized_random_forest_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "mlp_regressor_learner":
				self.processed_settings["learner"] = make_mlp_regressor_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "nonneg_rct_causal_random_forest_learner":
				self.processed_settings["learner"] = make_nonneg_rct_causal_random_forest_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "lassocv_selector_learner":
				self.processed_settings["learner"] = make_lassocv_selector_learner(random_seed=self.processed_settings["random_seed"])
			elif self.processed_settings["learner"] == "linreg_l2_learner":
				self.processed_settings["learner"] = linreg_l2_learner


	def _outcome_transform(self, X, Y, setting):
		'''
		For both of these, X will implicitly have the form (T, X) where T is treatment indicator
		Note that we are only ever considering propensities = 0.5
		'''
		if setting == "ipw_fair":
			pY = utils.causal_pseudo_outcome(X[:,0], X[:,1:], Y, 0.5)
		elif setting == "aipw_intercept_fair":
			rs = self.rng.randint(0, 2**32 - 1)
			pY = utils.aipw_intercept_pseudo_outcome(X[:,0], X[:,1:], Y, 0.5, random_seed=rs)
		else:
			raise ValueError("Invalid option for outcome_transform")
		return pY

	# ---------------------------------------------------- ABSTRACT FUNCTIONALITY ---------------------------------------------------- #

	def run_strategy(self, X, Y, meta={}):
		strategy = getattr(self, "_run_{}".format(self.processed_settings["strategy"]))
		rejected, region = strategy(X, Y, meta=meta)
		return rejected, region

	def simulate_batch(self):
		# Special routine for BART dataset
		if self.processed_settings["dgp"] == "BARTDataset":
			self.simulate_batch_bart()
			return

		# General simulations
		for i in range(self.processed_settings["n_sims"]):
			# Generate new random seed
			rs = self.rng.randint(0, 2**32 - 1)
			self.processed_settings["random_seed"] = rs
			# Sample
			sampler, base_sampler = getattr(self, "_generate_sampler_{}".format(self.processed_settings["dgp"]))()
			X, Y = sampler()
			if "outcome_transform" in self.processed_settings.keys():
				self.processed_settings["pY"] = self._outcome_transform(X, Y, self.processed_settings["outcome_transform"])
			else:
				self.processed_settings["pY"] = None
			# Special logic for NullDGPs
			if "SPECIAL_TOKEN_NULLDGPS" in self.processed_settings.keys() and self.processed_settings["SPECIAL_TOKEN_NULLDGPS"]:
				assert "binary" in self.processed_settings.keys() and "SPECIAL_TOKEN_NULLDGPS_TRANSFORM_TYPE" in self.processed_settings.keys(), "Since SPECIAL_TOKEN_NULLDGPS is active, we need binary and SPECIAL_TOKEN_NULLDGPS_TRANSFORM_TYPE settings to be defined"
				# If we activate this setting, we will retrieve the binary argument to see whether anything needs to be done
				# If so, we need self.processed_settings["SPECIAL_TOKEN_NULLDGPS_TRANSFORM_TYPE"] in ["ipw", "aipw"]
				# X, as given above, will in fact be TX. We apply the special transformation to produce pY, and then let (X, Y) <- (TX[:,1:], ipwY)
				# We then save self.processed_settings["pY"] = pY
				if not self.processed_settings["binary"]:
					ipwY = utils.causal_pseudo_outcome(X[:,0], X[:,1:], Y, 0.5)
					if self.processed_settings["SPECIAL_TOKEN_NULLDGPS_TRANSFORM_TYPE"] == "ipw":
						X, Y, pY = X[:,1:], ipwY, ipwY
					elif self.processed_settings["SPECIAL_TOKEN_NULLDGPS_TRANSFORM_TYPE"] == "aipw":
						aipwY = null_dgps_aipw_transform(X[:,0], X[:,1:], Y, random_seed=rs)
						X, Y, pY = X[:,1:], ipwY, aipwY
					else:
						raise ValueError("Invalid option for SPECIAL_TOKEN_NULLDGPS_TRANSFORM_TYPE")
					self.processed_settings["pY"] = pY
			rejected, region = self.run_strategy(X, Y, meta={})
			metrics = getattr(self, "_evaluate_region_{}".format(self.processed_settings["dgp"]))(region)
			utility, size, sate, utility_se, size_se, sate_se = metrics
			self.simulation_results_data.append([self.processed_settings["task_id"], i,
												 rejected, size, sate, utility,
												 size_se, sate_se, utility_se])
		self.simulation_results_df = pd.DataFrame(self.simulation_results_data, columns=["task_id", "sim_id", "rejected",
																						 "region_mass", "subgroup_mean",
																						 "subgroup_utility", "region_mass_se",
																						 "subgroup_mean_se", "subgroup_utility_se"])

	def simulate_batch_bart(self):
		# Initialize some new results variables
		self.region_membership_vecs = []
		self.train_indic_vecs = []
		# Initialize BART sampler
		bart_sampler = BARTDataset(bart_path=self.processed_settings["bart_path"],
								   random_seed=self.processed_settings["random_seed"])
		# Simulate
		for i in range(self.processed_settings["n_sims"]):
			# Generate new random seed
			rs = self.rng.randint(0, 2**32 - 1)
			self.processed_settings["random_seed"] = rs
			# Sample
			train_indics, test_indics = bart_sampler.generate_train_test_indics(self.processed_settings["n"])
			TX, Y = bart_sampler.sample(train_indics)
			# Construct pseudo-outcome
			pY = utils.aipw_intercept_pseudo_outcome(TX[:,0], TX[:,1:], Y, bart_sampler.get_propensity(), random_seed=rs)
			if self.processed_settings["strategy"] == "Chiseling" or ("bonf_strategy" in self.processed_settings.keys() and self.processed_settings["bonf_strategy"] == "Chiseling"):
				# Due to a discrepancy in how test_thresh gets used, we only center pseudo-outcomes if method is chiseling.
				# For other methods, test_thresh is utilized directly by the test
				pY = pY - self.processed_settings["test_thresh"]
			self.processed_settings["pY"] = pY
			# Run strategy
			rejected, region = self.run_strategy(TX, Y)
			# Calculate and save metrics
			metrics = bart_sampler.estimate_region_metrics(region, test_indics, test_thresh=self.processed_settings["test_thresh"])
			utility, size, sate, utility_se, size_se, sate_se = metrics
			self.simulation_results_data.append([self.processed_settings["task_id"], i,
												 rejected, size, sate, utility,
												 size_se, sate_se, utility_se])
			# Also calculate and save the subgroup membership vector and the training indicator vector, if requested
			if self.processed_settings["save_subgroup_membership"]:
				if region is not None:
					rs = self.rng.randint(0,2**32-1)
					unit_reg = UnitRegistrar(rs)
					regcov = unit_reg.register_units(bart_sampler.bart_df.iloc[:,1:].values)
					self.region_membership_vecs.append(region.in_region(regcov))
					self.train_indic_vecs.append(train_indics)
				else:
					self.region_membership_vecs.append(np.zeros(len(train_indics)).astype(bool))
					self.train_indic_vecs.append(train_indics)
		self.simulation_results_df = pd.DataFrame(self.simulation_results_data, columns=["task_id", "sim_id", "rejected",
																						 "region_mass", "subgroup_mean",
																						 "subgroup_utility", "region_mass_se",
																						 "subgroup_mean_se", "subgroup_utility_se"])
		if self.processed_settings["save_subgroup_membership"]:
			self.region_membership_df = pd.DataFrame(self.region_membership_vecs)
			self.region_membership_df["task_id"] = self.processed_settings["task_id"]
			self.region_membership_df["sim_id"] = np.arange(self.processed_settings["n_sims"])
			self.train_indic_df = pd.DataFrame(self.train_indic_vecs)
			self.train_indic_df["task_id"] = self.processed_settings["task_id"]
			self.train_indic_df["sim_id"] = np.arange(self.processed_settings["n_sims"])

	# ---------------------------------------------------- SPECIFIC BENCHMARKS ---------------------------------------------------- #

	def _generate_sampler_BasicLinearRCT(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(BasicLinearRCT))
		base_sampler = BasicLinearRCT(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_BasicLinearRCT(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(BasicLinearRCT))
		sampler = BasicLinearRCT(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_KangSchaferRCT(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(KangSchaferRCT))
		base_sampler = KangSchaferRCT(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_KangSchaferRCT(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(KangSchaferRCT))
		sampler = KangSchaferRCT(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_BinaryDesignRCT(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(BinaryDesignRCT))
		base_sampler = BinaryDesignRCT(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_BinaryDesignRCT(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(BinaryDesignRCT))
		sampler = BinaryDesignRCT(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_BasicBinaryRegression(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(BasicBinaryRegression))
		base_sampler = BasicBinaryRegression(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_BasicBinaryRegression(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(BasicBinaryRegression))
		sampler = BasicBinaryRegression(**sampler_settings)
		metrics = sampler.estimate_region_metrics(self.processed_settings["test_thresh"], region)
		return metrics

	def _generate_sampler_HeterogeneousLinearRCT(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(HeterogeneousLinearRCT))
		base_sampler = HeterogeneousLinearRCT(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_HeterogeneousLinearRCT(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(HeterogeneousLinearRCT))
		sampler = HeterogeneousLinearRCT(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_NonNegRCT(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(NonNegRCT))
		base_sampler = NonNegRCT(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_NonNegRCT(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(NonNegRCT))
		sampler = NonNegRCT(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_LinearNonNegRCT(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(LinearNonNegRCT))
		base_sampler = LinearNonNegRCT(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_LinearNonNegRCT(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(LinearNonNegRCT))
		sampler = LinearNonNegRCT(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_NullDGPs(self):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(NullDGPs))
		base_sampler = NullDGPs(**sampler_settings)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_NullDGPs(self, region):
		sampler_settings = subset_dict(self.processed_settings, get_function_arguments(NullDGPs))
		sampler = NullDGPs(**sampler_settings)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _generate_sampler_BenchTM(self):
		base_sampler = BenchTM(scenario=self.processed_settings["scenario"],
							   benchtm_path=self.processed_settings["benchtm_path"],
							   benchtm_cache=self.benchtm_cache,
							   random_seed=self.processed_settings["random_seed"])
		if self.benchtm_cache is None:
			self.benchtm_cache = (base_sampler.T, base_sampler.X, base_sampler.Y, base_sampler.opt_metrics)
		sampler = lambda: base_sampler.sample(self.processed_settings["n"])
		return sampler, base_sampler

	def _evaluate_region_BenchTM(self, region):
		sampler = BenchTM(scenario=self.processed_settings["scenario"],
						  benchtm_path=self.processed_settings["benchtm_path"],
						  benchtm_cache=self.benchtm_cache,
						  random_seed=self.processed_settings["random_seed"])
		if self.benchtm_cache is None:
			self.benchtm_cache = (sampler.T, sampler.X, sampler.Y, sampler.opt_metrics)
		metrics = sampler.estimate_region_metrics(region)
		return metrics

	def _run_Chiseling(self, X, Y, meta={}):
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(Chiseling))
		strategy = Chiseling(X=X, Y=Y, **strategy_settings)
		strategy.run_strategy()
		region = strategy.protocol.get_rejected_region()
		if "substitute_t_test" in self.processed_settings.keys() and self.processed_settings["substitute_t_test"]:
			if strategy.protocol.pY_masked.shape[0] >= 2:
				ttest_res = scipy.stats.ttest_1samp(strategy.protocol.pY_masked, popmean=0, alternative='greater')
				rejected = (ttest_res.pvalue <= self.processed_settings["alpha"])
			else:
				rejected = False
		else:
			rejected = (region is not None)
		return rejected, region

	def _run_DataSplittingStrategy(self, X, Y, meta={}):
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(DataSplittingStrategy))
		strategy = DataSplittingStrategy(X=X, Y=Y, **strategy_settings)
		strategy.run_strategy()
		region = strategy.region if strategy.rejected else None
		return strategy.rejected, region

	def _run_SimulDataSplittingStrategy(self, X, Y, meta={}):
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(SimulDataSplittingStrategy))
		strategy = SimulDataSplittingStrategy(X=X, Y=Y, **strategy_settings)
		strategy.run_strategy()
		region = strategy.region if strategy.rejected else None
		return strategy.rejected, region

	def _run_TTestStrategy(self, X, Y, meta={}):
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(TTestStrategy))
		strategy = TTestStrategy(Y=Y, **strategy_settings)
		strategy.run_strategy()
		region = strategy.region if strategy.rejected else None
		return strategy.rejected, region

	def _run_BonferroniCombiner(self, X, Y, meta={}):
		if self.processed_settings["bonf_strategy"] == "Chiseling":
			strategy_settings = subset_dict(self.processed_settings, get_function_arguments(Chiseling))
		elif self.processed_settings["bonf_strategy"] == "DataSplittingStrategy":
			strategy_settings = subset_dict(self.processed_settings, get_function_arguments(DataSplittingStrategy))
		elif self.processed_settings["bonf_strategy"] == "SimulDataSplittingStrategy":
			strategy_settings = subset_dict(self.processed_settings, get_function_arguments(SimulDataSplittingStrategy))
		else:
			raise ValueError("Invalid option for strategy")
		strategy = BonferroniCombiner(X=X,
									  Y=Y,
									  strategy=self.processed_settings["bonf_strategy"],
									  strategy_settings=strategy_settings,
									  alpha=self.processed_settings["alpha"],
									  random_seed=self.processed_settings["random_seed"])
		strategy.run_strategy()
		region = strategy.rejected_region
		rejected = (region is not None)
		return rejected, region
