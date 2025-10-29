import numpy as np
import scipy.stats

from ...protocol.IRST import IRST, IRSTBinary, UnitRegistrar
from .alpha_spending_strategy import AlphaSpendingStrategy
from .equal_alpha_strategy import EqualAlphaStrategy

class Chiseling:
	'''
	This is a wrapper that instantiates protocol and runs AlphaSpendingStrategy
	Note that WLOG non-binary chiseling tests mean <= 0. test_thresh only affects behavior of strategy class and binary chiseling.
	Thus, if binary = False, should have test_thresh = 0.
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
		self.X = X
		self.Y = Y
		self.test_thresh = test_thresh
		self.alpha = alpha
		self.learner = learner
		self.n_burn_in = n_burn_in
		self.pY = pY
		self.binary = binary
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
		rs = self.rng.randint(0, 2**32 - 1)
		self.strategy = AlphaSpendingStrategy(protocol=self.protocol,
				 							  test_thresh=self.test_thresh,
				 							  learner=self.learner,
				 							  n_burn_in=self.n_burn_in,
											  reveal_batch_size=self.reveal_batch_size,
											  refit_batch_size=self.refit_batch_size,
											  n_min=self.n_min,
											  alpha_min=self.alpha_min,
											  alpha_spending_fn=self.alpha_spending_fn,
											  boundary_strategy=self.boundary_strategy,
											  use_learner_weights=self.use_learner_weights,
											  skip_const_predictor=self.skip_const_predictor,
											  tiebreak=self.tiebreak,
											  quit_on_rejection=self.quit_on_rejection,
											  random_seed=rs)

	def run_strategy(self, verbose=False):
		self.strategy.run_strategy(verbose=verbose)



class ChiselingEqualAlpha:
	'''
	This is a wrapper that instantiates protocol and runs EqualAlphaStrategy
	Note that WLOG non-binary chiseling tests mean <= 0. test_thresh only affects behavior of strategy class and binary chiseling.
	Thus, if binary = False, should have test_thresh = 0.
	'''
	def __init__(self,
				 X,
				 Y,
				 test_thresh,
				 alpha,
				 learner,
				 pY=None,
				 binary=False,
				 num_tests=20,
				 n_burn_in="auto",
				 n_min='auto',
				 reveal_batch_size=1,
				 refit_batch_size=1,
				 use_learner_weights=False,
				 tiebreak=False,
				 random_seed=None):
		self.X = X
		self.Y = Y
		self.test_thresh = test_thresh
		self.alpha = alpha
		self.learner = learner
		self.pY = pY
		self.binary = binary
		self.num_tests = num_tests
		self.n_burn_in = n_burn_in
		self.n_min = n_min
		self.reveal_batch_size = reveal_batch_size
		self.refit_batch_size = refit_batch_size
		self.use_learner_weights = use_learner_weights
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
		rs = self.rng.randint(0, 2**32 - 1)
		self.strategy = EqualAlphaStrategy(protocol=self.protocol,
										   test_thresh=self.test_thresh,
										   learner=self.learner,
										   num_tests=self.num_tests,
										   n_burn_in=self.n_burn_in,
										   n_min=self.n_min,
										   reveal_batch_size=self.reveal_batch_size,
										   refit_batch_size=self.refit_batch_size,
										   use_learner_weights=self.use_learner_weights,
										   tiebreak=self.tiebreak,
										   random_seed=rs)

	def run_strategy(self, verbose=False):
		self.strategy.run_strategy(verbose=verbose)
