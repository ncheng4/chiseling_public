# ───────────────────────────── WRITTEN BY CHATGPT ─────────────────────────────

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Optional
from scipy.stats import norm

from .utility_tree import UDecisionTree


class URandomForest:
	"""
	Simple random-forest wrapper around UDecisionTree.
	"""

	def __init__(
		self,
		n_trees: int = 100,
		eps_min: float = 0.0,
		d_max: int = 3,
		m_try: int | None = 10,
		n_min: int = 10,
		alpha: float = 0.05,
		random_state: int | None = None,   # can be positional or keyword
		*,                                 # everything after * is keyword-only
		n_test: int,                       # still required
	):
		if n_trees <= 0:
			raise ValueError("n_trees must be positive")

		self.n_trees = n_trees
		self.tree_params = dict(
			eps_min=eps_min,
			d_max=d_max,
			m_try=m_try,
			n_min=n_min,
			alpha=alpha,
			n_test=n_test,           # keyword-only param
		)
		self._rng = np.random.default_rng(random_state)
		self._random_state0 = random_state   # keep the seed for repr/debug
		self.trees: List[UDecisionTree] = []

	# --------------------------------------------------------------------- #
	#                                FIT                                    #
	# --------------------------------------------------------------------- #
	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		sample_weight: Optional[np.ndarray] = None,
	) -> "URandomForest":

		X = np.asarray(X)
		y = np.asarray(y)
		n_samples = y.shape[0]
		if X.ndim != 2 or n_samples != X.shape[0]:
			raise ValueError("X must be (n_samples, n_features) and match y")

		if sample_weight is None:
			base_weights = np.ones(n_samples)
		else:
			base_weights = np.asarray(sample_weight, dtype=float)
			if base_weights.shape[0] != n_samples:
				raise ValueError("sample_weight length mismatch")
			if (base_weights < 0).any():
				raise ValueError("sample_weight must be non-negative")

		# normalise for weighted bootstrap
		probs = base_weights / base_weights.sum()

		self.trees = []
		for i in range(self.n_trees):
			# ---------------------------------------------------------- #
			# 1. draw bootstrap indices (weighted or uniform)            #
			# ---------------------------------------------------------- #
			indices = self._rng.choice(
				n_samples, size=n_samples, replace=True, p=probs
			)

			X_boot = X[indices]
			y_boot = y[indices]
			w_boot = None if sample_weight is None else base_weights[indices]

			# ---------------------------------------------------------- #
			# 2. make an independent tree                                #
			# ---------------------------------------------------------- #
			tree_seed = None if self._random_state0 is None else self._rng.integers(0, 2**32-1)
			tree = UDecisionTree(random_state=tree_seed, **self.tree_params)
			tree.fit(X_boot, y_boot, sample_weight=w_boot)
			self.trees.append(tree)

		return self

	# --------------------------------------------------------------------- #
	#                          PREDICTION                                   #
	# --------------------------------------------------------------------- #
	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		if not self.trees:
			raise RuntimeError("Call fit() before predict()")

		preds = np.column_stack([t.predict(X) for t in self.trees])
		return preds.mean(axis=1)          # shape (n_samples,)

	def predict(self, X: np.ndarray) -> np.ndarray:
		proba = self.predict_proba(X)
		return (proba >= 0.5).astype(int)

	# --------------------------------------------------------------------- #
	#                         REPRESENTATION                                #
	# --------------------------------------------------------------------- #
	def __repr__(self):
		return (
			f"URandomForest(n_trees={self.n_trees}, "
			f"tree_params={self.tree_params}, "
			f"seed={self._random_state0})"
		)