import numpy as np
import pandas as pd
import scipy.special
import scipy.stats

from .dgps import DGP
from ..source.protocol.IRST import UnitRegistrar

# -------------------------------- Sampler class -------------------------------- #

class BARTDataset(DGP):
	'''
	bart_path must point to a .tsv that has outcome as first column, treatment as second, and pre-processed covariates as remaining
	'''
	def __init__(self, bart_path, random_seed=None):
		super().__init__(random_seed=random_seed)
		self.bart_path = bart_path
		# Load data
		self.bart_df = pd.read_csv(self.bart_path, sep="\t")
		# Propensity
		self.propensity = self.bart_df.treatment.mean()

	def get_propensity(self):
		return self.propensity

	def generate_train_test_indics(self, n):
		train_inds = self.rng.choice(self.bart_df.shape[0], n, replace=False)
		train_indics = np.zeros(self.bart_df.shape[0]).astype(bool)
		train_indics[train_inds] = True
		test_indics = ~train_indics
		return train_indics, test_indics

	def sample(self, train_indics):
		train_bart_df = self.bart_df.iloc[train_indics]
		TX = train_bart_df.iloc[:,1:].values
		Y = train_bart_df.iloc[:,0].values
		return TX, Y

	def estimate_region_metrics(self, region, test_indics, test_thresh=0):
		if region is None:
			return 0, 0, np.nan, 0, 0, np.nan
		# Sample
		TX, Y = self.sample(test_indics)
		T = TX[:,0]
		# Construct pseudo-outcome and subtract test_thresh
		pY = (Y * T) / self.propensity  - (Y * (1 - T)) / (1 - self.propensity)
		pY = pY - test_thresh
		# Register units and check subgroup membership
		rs = self.rng.randint(0,2**32-1)
		unit_reg = UnitRegistrar(rs)
		regcov = unit_reg.register_units(TX)
		subgroup_inds = region.in_region(regcov)
		# Estimate region metrics
		util = (pY * subgroup_inds).mean()
		util_se = np.sqrt((pY * subgroup_inds).var() / pY.shape[0])
		size = subgroup_inds.mean()
		size_se = np.sqrt(subgroup_inds.var() / pY.shape[0])
		subgroup_scores = pY[subgroup_inds]
		if subgroup_scores.shape[0] >= 2:
			sub_mean = subgroup_scores.mean()
			sub_mean_se = np.sqrt(subgroup_scores.var() / subgroup_scores.shape[0])
		else:
			sub_mean = np.nan
			sub_mean_se = np.nan
		return util, size, sub_mean, util_se, size_se, sub_mean_se

