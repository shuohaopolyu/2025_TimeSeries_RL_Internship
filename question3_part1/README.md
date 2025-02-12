# Dynamic Causal Bayesian Optimization

## Overview
This folder contains the implementation of the Dynamic Causal Bayesian Optimization (DCBO) method as proposed by Aglietti et al. It leverages the TensorFlow Probability framework for conducting analyses. The folder is structured as follows:

- `causal_graph/`: Contains methods for generating and processing causal graphs.
- `demo/`: Contains demonstrations of question 1 and synthetic experiments.
- `methods/`: Implementation of the dynamic causal Bayesian optimization method.
- `sem/`: Includes implementations of various structural equation models.
- `tests/`: Contains unit and integration tests to ensure the reliability of the code.
- `utils/`: Useful functions for data processing in the context of the DCBO method.
- `report_cdbo.pdf`: Discusses the implementation of the DCBO method.


## Demonstrations
We offer three demonstrations to illustrate the capabilities of our implementation:

* **[Demo 1. Exploration set identification:](demo/demo1_find_exploration_set.ipynb)** Demonstrates how to find an exploration set for a given causal graph and target variable.
* **[Demo 2a. Stationary DAG and SCM experiment replication:](demo/demo2a_stat.ipynb)** Replicates the first experiment in the paper, where a stationary DAG and SCM are used to evaluate the DCBO method.
* **[Demo 2b. Noisy manipulative variables:](demo/demo2b_noisy.ipynb)** Demonstrates the DCBO method's performance when the noise level of variables is increased.

Below is an animation that provides a quick overview of how DCBO operates:


![My Animated GIF](demo/experiments/dcbp.gif)


## Prerequisites
- Python >= 3.8
- TesnorFlow = 2.18.0
- TensorFlow-Probability = 0.25.0
- NetworkX = 3.4.2
