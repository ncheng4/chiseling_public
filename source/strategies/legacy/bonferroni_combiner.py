import numpy as np
from .chiseling import Chiseling
from ..data_splitting_strategy import DataSplittingStrategy
from ..simul_data_splitting_strategy import SimulDataSplittingStrategy

class BonferroniCombiner:
	def __init__(self,
				 X,
				 Y,
				 strategy,
				 strategy_settings,
				 alpha,
				 splits=[0.2,0.5,0.8],
				 random_seed=None):
		self.X = X
		self.Y = Y
		self.strategy = strategy
		self.strategy_settings = dict(strategy_settings)
		self.alpha = alpha
		self.splits = splits
		self.rng = np.random.RandomState(random_seed)
		# Initialize
		self.strategy_objs = []
		self._initialize_strategy()
		# Placeholder for results
		self.split_rejections = []
		self.split_regions = []
		self.rejected_region_split = None
		self.rejected_region = None

	def _initialize_strategy(self):
		for p in self.splits:
			rs = self.rng.randint(0, 2**32 - 1)
			strategy_settings = dict(self.strategy_settings)
			strategy_settings["random_seed"] = rs
			strategy_settings["alpha"] = self.alpha / len(self.splits)
			if self.strategy == "Chiseling":
				strategy_settings["n_burn_in"] = p
				strategy_obj = Chiseling(X=self.X, Y=self.Y, **strategy_settings)
			elif self.strategy == "DataSplittingStrategy":
				strategy_settings["train_ratio"] = p
				strategy_obj = DataSplittingStrategy(X=self.X, Y=self.Y, **strategy_settings)
			elif self.strategy == "SimulDataSplittingStrategy":
				strategy_settings["train_ratio"] = p
				strategy_obj = SimulDataSplittingStrategy(X=self.X, Y=self.Y, **strategy_settings)
			else:
				raise ValueError("Invalid option for strategy. Must be among {AlphaSpendingStrategy, DataSplittingStrategy, SimulDataSplittingStrategy}.")
			self.strategy_objs.append(strategy_obj)

	def run_strategy(self):
		for strategy_obj in self.strategy_objs:
			strategy_obj.run_strategy()
			if self.strategy == "Chiseling":
				region = strategy_obj.protocol.get_rejected_region()
				rejected = (region is not None)
				self.split_rejections.append(rejected)
				self.split_regions.append(region)
			else:
				region = strategy_obj.region if strategy_obj.rejected else None
				self.split_rejections.append(strategy_obj.rejected)
				self.split_regions.append(region)
		# Choose rejection corresponding to largest p
		rejected_splits = np.array(self.splits)[self.split_rejections]
		if len(rejected_splits) > 0:
			largest_split = max(rejected_splits)
			sel_ind = np.where(np.array(self.splits) == largest_split)[0][0]
			self.rejected_region_split = largest_split
			self.rejected_region = self.split_regions[sel_ind]
