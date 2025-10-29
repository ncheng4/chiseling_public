# Chiseling

This is the codebase for results found in ["Chiseling: Powerful and Valid Subgroup Selection via Interactive Machine Learning" (Cheng, Spector, Janson 2025)](https://arxiv.org/abs/2509.19490).

Abstract chiseling functionality for shrinking and testing is available under ```source/protocol/IRST.py```. Specific instantiations of chiseling (including those used to obtain results in the paper) are available under ```source/strategies/```. Machine learning methods used may be found under ```source/learners/```.

Data generating processes are available under the ```dgps/``` folder, while the methods used to execute the simulations used in the paper are under ```benchmark/``` and ```scripts/benchmark```. Specific configurations for the simulations can be found at ```notebooks/simulation_setup/```.

The re-analysis of the GSS survey experiment can be found under ```notebooks/bart_analysis/```.

More documentation to come.
