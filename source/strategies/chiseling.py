import numpy as np
import scipy.stats

from ..protocol.IRST import IRST, IRSTBinary, UnitRegistrar
from .simple_margin_strategy import SimpleMarginStrategy


class Chiseling:
	'''
	This is a wrapper that instantiates protocol and runs SimpleMarginStrategy
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
		self.strategy = SimpleMarginStrategy(protocol=self.protocol,
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
				 							 tiebreak=self.tiebreak)

	def run_strategy(self, verbose=False):
		self.strategy.run_strategy(verbose=verbose)

