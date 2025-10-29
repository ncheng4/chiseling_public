# ───────────────────────────── WRITTEN BY CHATGPT ─────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import norm


class UDecisionTree:
	# ───────────────────────────── internal node ──────────────────────────────
	@dataclass
	class _Node:
		feature: int | None
		threshold: float | None
		label: int                        # 0 or 1
		left: UDecisionTree._Node | None = None
		right: UDecisionTree._Node | None = None

		def is_leaf(self) -> bool:
			return self.feature is None

	# ───────────────────────────── constructor ────────────────────────────────
	def __init__(
		self,
		eps_min: float = 0.0,
		d_max: int = 3,
		m_try: int | None = 10,
		n_min: int = 10,
		alpha: float = 0.05,
		random_state: int | None = None,
		*,                                   # everything after * is keyword-only
		n_test: int,
	):
		self.eps_min, self.d_max, self.m_try = eps_min, d_max, m_try
		self.n_min, self.alpha, self.n_test = n_min, alpha, n_test
		self._z = norm.ppf(1 - alpha)
		self._rng = np.random.default_rng(random_state)

		self._root: UDecisionTree._Node | None = None
		self._X: np.ndarray | None = None
		self._y: np.ndarray | None = None
		self._w: np.ndarray | None = None

	# ───────────────────────────── public API ────────────────────────────────
	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		sample_weight: Optional[np.ndarray] = None,
	) -> "UDecisionTree":
		X, y = np.asarray(X), np.asarray(y)
		if X.ndim != 2:
			raise ValueError("X must be 2-D (n_samples, n_features)")
		if y.shape[0] != X.shape[0]:
			raise ValueError("X and y must have the same number of rows")

		if sample_weight is None:
			w = np.ones_like(y, dtype=float)
		else:
			w = np.asarray(sample_weight, dtype=float)
			if w.shape[0] != y.shape[0]:
				raise ValueError("sample_weight length mismatch")

		self._X, self._y, self._w = X, y, w

		n = y.size
		assign = np.ones(n, dtype=bool)          # P = all samples (label 1)
		mask = np.ones(n, dtype=bool)

		self._root = self._Node(None, None, 1)
		self._grow(self._root, mask, assign, depth=0)
		return self

	def predict(self, X_new: np.ndarray) -> np.ndarray:
		if self._root is None:
			raise RuntimeError("Call fit() before predict().")

		X_new = np.asarray(X_new)
		if X_new.ndim == 1:
			X_new = X_new.reshape(1, -1)

		out = np.empty(X_new.shape[0], dtype=int)
		for i, x in enumerate(X_new):
			node = self._root
			while not node.is_leaf():
				node = node.left if x[node.feature] <= node.threshold else node.right
			out[i] = node.label
		return out

	# ───────────────────────────── recursive build ───────────────────────────
	def _grow(
		self,
		node: _Node,
		node_mask: np.ndarray,
		assign: np.ndarray,
		depth: int,
	):
		if depth >= self.d_max:
			return

		split = self._best_split(node_mask, assign)
		if split is None:
			return

		feat, thr, lmask, rmask, llbl, rlbl, _ = split
		node.feature, node.threshold = feat, thr
		node.left = self._Node(None, None, llbl)
		node.right = self._Node(None, None, rlbl)

		new_assign = assign.copy()
		new_assign[lmask] = llbl
		new_assign[rmask] = rlbl

		self._grow(node.left, lmask, new_assign, depth + 1)
		self._grow(node.right, rmask, new_assign, depth + 1)

	# ───────────────────────────── split search ──────────────────────────────
	def _best_split(self, node_mask: np.ndarray, assign: np.ndarray):
		X, y, w = self._X, self._y, self._w
		cur_U = self._utility(assign)

		best_delta, best = -np.inf, None

		for j in range(X.shape[1]):
			col = X[node_mask, j]
			if col.size < 2 * self.n_min:
				continue

			for thr in self._thresholds(col):
				lmask = node_mask & (X[:, j] <= thr)
				rmask = node_mask & ~lmask
				if lmask.sum() < self.n_min or rmask.sum() < self.n_min:
					continue

				# only (1,0) and (0,1)
				for llbl, rlbl in ((1, 0), (0, 1)):
					new_assign = assign.copy()
					new_assign[lmask] = llbl
					new_assign[rmask] = rlbl

					delta = self._utility(new_assign) - cur_U
					if delta > best_delta:
						best_delta = delta
						best = (j, thr, lmask, rmask, llbl, rlbl, delta)

		if best is None or best_delta < self.eps_min:
			return None
		return best

	def _thresholds(self, col: np.ndarray) -> np.ndarray:
		uniq = np.unique(col)
		if uniq.size <= 1:
			return np.array([], dtype=col.dtype)
		m = self.m_try or uniq.size - 1
		qs = np.linspace(0, 1, m + 2)[1:-1]
		return np.unique(np.quantile(uniq, qs))

	# ───────────────────────────── utility U(P) ──────────────────────────────
	def _utility(self, assign: np.ndarray) -> float:
		P = assign
		w, y = self._w, self._y

		w_P = w[P]
		w_total = w.sum()
		if w_P.sum() == 0 or w_total == 0:
			return -np.inf

		rho = w_P.sum() / w_total                        # weighted proportion
		mu = np.dot(w_P, y[P]) / w_P.sum()               # weighted mean
		var = np.dot(w_P, (y[P] - mu) ** 2) / w_P.sum()  # weighted variance
		sigma = np.sqrt(var)

		if rho == 0 or sigma == 0 or not np.isfinite([rho, mu, sigma]).all():
			return -np.inf

		power_arg = np.sqrt(self.n_test * rho) * mu / sigma - self._z
		return mu * rho * norm.cdf(power_arg)