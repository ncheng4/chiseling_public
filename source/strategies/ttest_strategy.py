import numpy as np
import scipy.stats

from ..protocol.IRST import Region

class TTestStrategy:
	def __init__(self,
				 Y,
				 test_thresh,
				 alpha,
				 pY=None):
		self.Y = Y
		self.test_thresh = test_thresh
		self.alpha = alpha
		self.pY = pY
		# Set defaults
		if self.pY is None:
			self.pY = Y
		# Result variables
		self.region = Region()
		self.teststat = None
		self.pval = None
		self.rejected = None

	def run_strategy(self):
		res = scipy.stats.ttest_1samp(self.pY, popmean=self.test_thresh, alternative="greater")
		self.teststat, self.pval = res.statistic, res.pvalue
		self.rejected = (self.pval <= self.alpha)
