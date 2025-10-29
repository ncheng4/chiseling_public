# ───────────────────────────── WRITTEN BY CHATGPT ─────────────────────────────

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_X_y, check_array


class EfficientPOForest:
	"""
	Constructs semiparametric-efficient pseudo-outcomes (ψ) with
	default Random-Forest outcome models and a *constant* propensity
	ê = mean(W).

	After `fit(X, W, y)` the vector `psi_` is available.

	Parameters
	----------
	rf_kwargs : dict, optional
		Extra hyper-parameters forwarded to both internal
		RandomForestRegressor models (mu1 & mu0).
	random_state : int or None
		Seed passed on to the forests.
	"""

	def __init__(self, rf_kwargs: dict | None = None, random_state: int | None = None):
		self.rf_kwargs = rf_kwargs or {}
		self.random_state = random_state

		self.mu1_: RandomForestRegressor | None = None
		self.mu0_: RandomForestRegressor | None = None
		self.e_hat_: float | None = None
		self.psi_: np.ndarray | None = None

	# ------------------------------------------------------------------ #
	#                                FIT                                 #
	# ------------------------------------------------------------------ #
	def fit(self, X, W, y):
		"""
		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
		W : array-like, shape (n_samples,)  {0,1}
		y : array-like, shape (n_samples,)
		"""
		X, y = check_X_y(X, y, accept_sparse=False)
		W = np.asarray(W, dtype=int).ravel()
		if set(np.unique(W)) - {0, 1}:
			raise ValueError("W must be binary 0/1")

		treated = W == 1
		control = ~treated
		if treated.sum() == 0 or control.sum() == 0:
			raise ValueError("Need both treated and control observations.")

		# outcome forests
		self.mu1_ = RandomForestRegressor(random_state=self.random_state, **self.rf_kwargs)
		self.mu1_.fit(X[treated], y[treated])

		self.mu0_ = RandomForestRegressor(random_state=self.random_state, **self.rf_kwargs)
		self.mu0_.fit(X[control], y[control])

		# constant propensity
		self.e_hat_ = W.mean()
		if self.e_hat_ in (0.0, 1.0):
			raise ValueError("Constant ê is 0 or 1 – cannot build doubly-robust score.")

		# predictions
		mu1_hat = self.mu1_.predict(X)
		mu0_hat = self.mu0_.predict(X)
		mu_w_hat = np.where(treated, mu1_hat, mu0_hat)

		# efficient pseudo-outcome
		residual = y - mu_w_hat
		coeff = (W - self.e_hat_) / (self.e_hat_ * (1.0 - self.e_hat_))
		self.psi_ = coeff * residual + (mu1_hat - mu0_hat)
		return self

	# ------------------------------------------------------------------ #
	#                       ACCESSORS / UTILITIES                        #
	# ------------------------------------------------------------------ #
	def pseudo_outcome(self) -> np.ndarray:
		"""Return ψ (alias for `psi_`)."""
		if self.psi_ is None:
			raise RuntimeError("Call fit() first.")
		return self.psi_

	# convenience: let people write .predict(X) if they like ------------
	def predict(self, X):
		"""Alias that just re-computes μ1̂-μ0̂ (no weighting)."""
		if self.mu1_ is None or self.mu0_ is None:
			raise RuntimeError("Call fit() first.")
		X = check_array(X, accept_sparse=False)
		return self.mu1_.predict(X) - self.mu0_.predict(X)

	# nicely printable
	def __repr__(self):
		return (
			f"EfficientPOForest(rf_kwargs={self.rf_kwargs}, "
			f"random_state={self.random_state})"
		)