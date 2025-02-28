# Efficient Bayesian Inference with Latent Hamiltonian Neural Networks in No-U-Turn Sampling
## Overview
This repository contains the replication code for the method in the paper "Efficient Bayesian Inference with Latent Hamiltonian Neural Networks in No-U-Turn Sampling" by Dhulipala et al. The code is based on TensorFlow 2.0 framework, and is organized as follows:
- `demo/`: Contains five demo scripts to run the experiments in the paper.
- `exps/`: Includes the necessary weights data for demo scripts.
- `hamilton_neural_network/`: Contains the implementation of the Hamiltonian neural network and Latent Hamiltonian neural network model.
- `hamilton_system/`: Contains the implementation of the Hamiltonian System based on the given probability model.
- `no_u_turn/`: Includes the implementation of the Efficient No-U-Turn Sampling algorithm.
- `pdf_models/`: Contains the implementation of the probability models used in the experiments.
- `tests/`: Includes the unit tests for all the modules.
- `report_lhnn_nuts.pdf`: Answers to sub-question b.

## Answer to sub-question d
The `mcmc` in TensorFlow-Probabilty is a powerful tool that encompasses a broad range of Markov chain Monte Carlo algorithms, in which Hamiltonian Monte Carlo and the necessary No-U-Turn sampling are included as well. However, the replication code for the paper does not utilize the TensorFlow-Probability library. Instead, the code is implemented from scratch using TensorFlow 2.0. The primary reason for this is that the Latent Hamiltonian neural network is deeply integrated into various components of the proposed method, including the internal algorithms of Hamiltonian Monte Carlo and No-U-Turn sampling. As a result, it is more practical to implement the entire method from scratch, allowing for greater flexibility and customization.

## Prerequisites
- Python >= 3.8
- TensorFlow = 2.18.0