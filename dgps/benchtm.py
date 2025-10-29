import numpy as np
import pandas as pd
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

# -------------------------------- Helper functions -------------------------------- #

def one_hot_encode_non_numeric(df):
	"""
	One-hot encodes non-numeric columns in a DataFrame.

	Parameters:
	df (pd.DataFrame): Input DataFrame with mixed data types.

	Returns:
	pd.DataFrame: DataFrame with all non-numeric columns one-hot encoded.
	"""
	# Identify non-numeric columns
	non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
	
	# Perform one-hot encoding on non-numeric columns
	df_encoded = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)
	
	return df_encoded

# -------------------------------- Sampler class -------------------------------- #

class BenchTM(DGP):
	'''
	benchtm_path must point to a path that has files benchtm.processed.{SCENARIO}.tsv and benchtm_opt_metrics.tsv
	If benchtm_cache provided, must constitute a tuple (T, X, Y, opt_metrics)

	Currently only supports scenarios 1 through 20
	There are 4 models and 5 heterogeneity levels. Scenarios 1-5 correspond to first model, increasing heterogeneity, and so on

	Bernoulli(0.5) experiment
	'''
	def __init__(self, scenario, benchtm_path, benchtm_cache=None, random_seed=None):
		super().__init__(random_seed=random_seed)
		assert 1 <= scenario <= 20, "Currently only supports scenarios 1 through 20"
		self.scenario = scenario
		self.benchtm_path = benchtm_path
		self.benchtm_cache = benchtm_cache
		# Load data
		if self.benchtm_cache is None:
			self.benchtm_df = pd.read_csv(self.benchtm_path + "/benchtm.processed.{}.tsv".format(self.scenario), sep="\t")
			self.opt_metrics = pd.read_csv(self.benchtm_path + "/benchtm_opt_metrics.tsv", sep="\t", index_col=0).loc[self.scenario]
			# Munge data
			self.T = self.benchtm_df["trt"].values.astype(float)
			X_raw = self.benchtm_df.loc[:,[col for col in self.benchtm_df.columns.values if col[0] == "X"]]
			self.X = one_hot_encode_non_numeric(X_raw).values.astype(float)
			self.Y = self.benchtm_df["Y"].values.astype(float)
		else:
			self.T, self.X, self.Y, self.opt_metrics = self.benchtm_cache
		# Store optimal utility estimates
		self.opt_utility_est = None
		self.opt_utility_se = None
		self.opt_reg_size = None
		self.opt_reg_size_se = None
		self.opt_reg_ate = None
		self.opt_reg_ate_se = None

	def sample(self, n):
		inds = self.rng.choice(self.Y.shape[0], n, replace=True)
		X = self.X[inds]
		T = self.T[inds]
		Y = self.Y[inds]
		TX = np.hstack((T.reshape(-1,1), X))
		return TX, Y

	def get_optimal_region_metrics(self):
		utility, size, sate = self.opt_metrics["utility"], self.opt_metrics["size"], self.opt_metrics["subgroup_mean"]
		utility_se, size_se, sate_se = 0, 0, 0
		return utility, size, sate, utility_se, size_se, sate_se

	def estimate_region_metrics(self, region, n_reps=100000):
		if region is None:
			return 0, 0, np.nan, 0, 0, np.nan
		# Sample
		TX, Y = self.sample(n_reps)
		T = TX[:,0]
		pY = 2 * Y * T - 2 * Y * (1 - T)
		# Register units and check subgroup membership
		rs = self.rng.randint(0,2**32-1)
		unit_reg = UnitRegistrar(rs)
		regcov = unit_reg.register_units(TX)
		subgroup_inds = region.in_region(regcov)
		# Estimate region metrics
		util = (pY * subgroup_inds).mean()
		util_se = np.sqrt((pY * subgroup_inds).var() / n_reps)
		size = subgroup_inds.mean()
		size_se = np.sqrt(subgroup_inds.var() / n_reps)
		subgroup_scores = pY[subgroup_inds]
		if subgroup_scores.shape[0] >= 2:
			sub_mean = subgroup_scores.mean()
			sub_mean_se = np.sqrt(subgroup_scores.var() / subgroup_scores.shape[0])
		else:
			sub_mean = np.nan
			sub_mean_se = np.nan
		return util, size, sub_mean, util_se, size_se, sub_mean_se

