import numpy as np
import pandas as pd
import argparse

from chiseling.benchmark.benchmark import Benchmark

def get_task_settings():
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--task-table-path', type=str, required=True)
	parser.add_argument('--task-ind', type=int, required=True)
	parser.add_argument('--saveprefix', type=str, required=True)
	args = parser.parse_args()
	# Load data; note that we take task-ind to be one-indexed, and must shift to align with zero-indexing
	TASK_IND = args.task_ind - 1
	SAVEPREFIX = args.saveprefix
	task_df = pd.read_csv(args.task_table_path, sep="\t", index_col=False)
	task_settings = task_df.iloc[TASK_IND]
	# Convert task to dict
	task_settings = task_settings.to_dict()
	return TASK_IND, task_settings, SAVEPREFIX

def run_sims():
	# Get task settings
	TASK_IND, task_settings, SAVEPREFIX = get_task_settings()

	# Set random seed
	np.random.seed(task_settings["random_seed"])

	# Simulate
	benchmark = Benchmark(task_settings)
	benchmark.simulate_batch()
	res_df = benchmark.simulation_results_df
	
	# Save data
	res_df.to_csv(SAVEPREFIX + ".{}.tsv".format(task_settings["task_id"]), sep="\t", index=False)

if __name__ == "__main__":
	run_sims()