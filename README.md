# 2025_TimeSeries_RL_Internship

This repository contains the code and documentation developed in response to the 6-month Off-Cycle AI & Data Science Associate Internship Program at the Machine Learning Center of Excellence in Hong Kong.

## Repository Structure
1. **question3_part1** (Completed on 9th January 2025)
    + Code and implementation for Dynamic Cusal Bayesian Optimization (DCBO) by Aglietti et al. (2021).
    + A demonstration of how to find the exploration set (Point 1 of Part 1), see [Demo 1. Exploration set identification](question3_part1/demo/demo1_find_exploration_set.ipynb).
    + Demonstrations of the programmatic implementation of DCBO, see [Demo 2a. Stat](question3_part1/demo/demo2a_stat.ipynb) and [Demo 2b. Noisy](question3_part1/demo/demo2b_noisy.ipynb).
    + A report for Point 3 of Part 1, see [Report_DCBO](question3_part1/report_dcbo.pdf).
    + Tests for the implementation, see the [tests](question3_part1/tests) folder.

2. **question3_part2** (Completed on 12th February 2025)
   + Code and implementation for the Bayesian Intervention Optimizatioon for Causal Discovery (BIOCD) by Wang et al. (2024).
   + A demonstration of causal discovery using BIOCD, see [Demo 1. Three SEMs](question3_part2/demo/demo1_three_sems.ipynb).
   + A report responding to points a-f of Part 2, see [Report_BIOCD](question3_part2/report_biocd.pdf).
   + Tests for the implementation, see [tests](question3_part2/tests) folder.

3. **question1_part1** (Completed on 3 March 2025)
   + Code and implementation for Efficient Bayesian Inference with Latent Hamiltonian Neural Networks in No-U-Turn Sampling by Dhulipala et al. (2023).
   + Demonstrations of the implementation, see [Demo 1. Gaussian Mixture Denxity](question1_part1/demo/demo1_gaussian_mixture_density_1d.ipynb), [Demo 2. 3-D Rosenbrock](question1_part1/demo/demo2_rosenbrock_density_3d.ipynb), [Demo 3. 2-D Neal’s Funnel](question1_part1/demo/demo3_neal_funnel_density_2d.ipynb), [Demo 4. 5-D ill-conditioned Gaussian](question1_part1/demo/demo4_ill_conditioned_gaussian_5d.ipynb), and [Demo 5.  10-D degenerate Rosenbrock](question1_part1/demo/demo5_rosenbrock_density_10d.ipynb).
   + A report for Part 1, see [Report_LHNN_NUTS](question1_part1/report_lhnn_nuts.pdf).
   + Tests for the implementation, see [tests](question1_part1/tests) folder.

4. **reports**
   + Individual reports for each part.

## Technical Specifications
+ Python 3.8 + TensorFlow 2.0 environment.
+ Comprehensive documentation and code comments.
+ Unit tests for all implementations using `unittest`.
