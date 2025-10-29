import numpy as np
import pandas as pd
import scipy.stats
import inspect
import warnings

import chiseling.source.protocol.utils as utils
from ..source.protocol.IRST import IRST, IRSTBinary, UnitRegistrar
from ..source.strategies.chiseling import Chiseling, ChiselingEqualAlpha
from ..source.strategies.data_splitting_strategy import DataSplittingStrategy
from ..source.strategies.simul_data_splitting_strategy import SimulDataSplittingStrategy
from ..source.strategies.bonferroni_combiner import BonferroniCombiner
from ..source.strategies.ttest_strategy import TTestStrategy
from ..source.learners.baselearners_general import linreg_learner, ridgecv_learner, make_lassocv_learner, make_elasticnetcv_learner, make_random_forest_learner
from ..source.learners.baselearners_binary import logreg_learner, logregl1_learner, make_logregcv_learner
from ..source.learners.baselearners_causal import causal_linreg_learner, make_causal_random_forest_learner

from ..dgps.basic_linear_rct import BasicLinearRCT
from ..dgps.binary_design_rct import BinaryDesignRCT
from ..dgps.kang_schafer_rct import KangSchaferRCT
from ..dgps.basic_binary_regression import BasicBinaryRegression
from ..dgps.benchtm import BenchTM

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
		for int_param in ["n_min", "n_burn_in", "reveal_batch_size", "refit_batch_size"]:
			if int_param in self.processed_settings.keys() and not np.isnan(self.processed_settings[int_param]):
				if int_param == "n_burn_in" and (0 < self.processed_settings[int_param] < 1):
					pass
				else:
					self.processed_settings[int_param] = int(self.processed_settings[int_param])
		# If strategy is oracle, we double the sample size and set the strategy to 50/50 data splitting
		if self.processed_settings["strategy"] == "OracleStrategy":
			self.processed_settings["strategy"] = "DataSplittingStrategy"
			self.processed_settings["train_ratio"] = 0.5
			self.processed_settings["n"] = 2 * self.processed_settings["n"]
		# Get learner from string option
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
			rejected, region = self.run_strategy(X, Y, meta={"base_sampler": base_sampler})
			metrics = getattr(self, "_evaluate_region_{}".format(self.processed_settings["dgp"]))(region)
			utility, size, sate, utility_se, size_se, sate_se = metrics
			self.simulation_results_data.append([self.processed_settings["task_id"], i,
												 rejected, size, sate, utility,
												 size_se, sate_se, utility_se])
		self.simulation_results_df = pd.DataFrame(self.simulation_results_data, columns=["task_id", "sim_id", "rejected",
																						 "region_mass", "subgroup_mean",
																						 "subgroup_utility", "region_mass_se",
																						 "subgroup_mean_se", "subgroup_utility_se"])

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
			if "alpha_spending_fn" not in self.processed_settings.keys() or self.processed_settings["alpha_spending_fn"] != "instantaneous":
				warnings.warn("Detected substitute_t_test=True but alpha_spending_fn not set to instantaneous. This may not produce valid behavior.")
			if protocol.pY_masked.shape[0] >= 2:
				ttest_res = scipy.stats.ttest_1samp(strategy.protocol.pY_masked, popmean=0, alternative='greater')
				rejected = (ttest_res.pvalue <= self.processed_settings["alpha"])
			else:
				rejected = False
		else:
			rejected = (region is not None)
		return rejected, region

	def _run_ChiselingEqualAlpha(self, X, Y, meta={}):
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(ChiselingEqualAlpha))
		strategy = ChiselingEqualAlpha(X=X, Y=Y, **strategy_settings)
		strategy.run_strategy()
		region = strategy.protocol.get_rejected_region()
		rejected = (region is not None)
		return rejected, region

	def _run_DataSplittingStrategy(self, X, Y, meta={}):
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(DataSplittingStrategy))
		strategy = DataSplittingStrategy(X=X, Y=Y, **strategy_settings)
		strategy.run_strategy()
		region = strategy.region if strategy.rejected else None
		return strategy.rejected, region

	def _run_SimulDataSplittingStrategy(self, X, Y, meta={}):
		if "binary" in self.processed_settings.keys() and self.processed_settings["binary"]:
			pY1X = meta["base_sampler"].calculate_probs(X)
		else:
			pY1X = None
		strategy_settings = subset_dict(self.processed_settings, get_function_arguments(SimulDataSplittingStrategy))
		strategy = SimulDataSplittingStrategy(X=X, Y=Y, pY1X=pY1X, **strategy_settings)
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
			if "binary" in self.processed_settings.keys() and self.processed_settings["binary"]:
				pY1X = meta["base_sampler"].calculate_probs(X)
				strategy_settings["pY1X"] = pY1X
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
