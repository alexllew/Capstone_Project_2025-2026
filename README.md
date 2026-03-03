This repository contains my work for a Bayesian Black-Box Optimisation (BBO) capstone project. The challenge simulates a realistic optimisation setting in which objective functions are unknown, expensive to evaluate, and sparsely observed, optimising eight different black-box functions of increasing dimensionality from two to eight input variables.

Section 1 - Project Overview

The goal of the project was to identify high-performing input values with very few function evaluations. This reflects many real-world machine learning and scientific optimisation problems, such as hyperparameter tuning and drug discovery, where evaluations are costly/time-consuming.

The project emphasises reasoned decision-making under uncertainty, careful modelling assumptions, and iterative refinement of strategy as new data arrive, as opposed to the final solution obtained.

This project demonstrates my ability to:

    Work with limited and noisy data
    Select and justify appropriate modelling assumptions and strategies
    Communicate technical decisions through code and documentation

Section 2 - Inputs and Outputs

Each optimisation task is defined by an unknown black-box function that maps inputs to outputs

Inputs

The input vectors were provided as NumPy arrays, with each row corresponding to a single query. Dimensionality varied by function:

    Function 1–2: 2D
    Function 3: 3D
    Function 4–5: 4D
    Function 6: 5D
    Function 7: 6D
    Function 8: 8D

All input values were continuous and constrained to the range [0, 1]

Outputs

Each output was a single scalar value representing performance, score, or utility. All tasks were framed as maximisation problems (such as negative loss)

Section 3 - Challenge Objectives

The objective of the BBO capstone project was to maximise the output of each black-box function using a very limited number of queries.

Key constraints include:

    A small initial dataset (10–40 points)
    Only one new query allowed per week
    Unknown functional form, noise level, and smoothness
    Increasing dimensionality across tasks

Because the functions were unknown and potentially non-linear with multiple local optima, brute-force or grid-based optimisation was infeasible. Instead, the challenge was to make informed, strategic choices about where to sample next, learning from previous observations while managing uncertainty. An important part of the challenge is the quality and justification of the optimisation strategy, not just the final value achieved.

Section 4 - Technical Approach

Overview

My approach evolved across the first three query submissions as I adapted to dimensionality, data scarcity, observed noise and smoothness, and signs of model over- or under-fitting. Rather than applying a single method blindly, I deliberately adjusted model complexity and acquisition strategy to match what the data can realistically support.

Initial Strategy

I used Gaussian Process (GP) regression as a probabilistic surrogate model. GPs are well-suited to black-box optimisation because they provide uncertainty estimates as well as point predictions, work well with small datasets, and naturally support probabilistic acquisition functions such as Upper Confidence Bound (UCB).

I experimented with multiple kernels (RBF, Matérn 1.5, Matérn 2.5, Rational Quadratic) and compared them using cross-validated performance and diagnostic plots. Rather than trusting a single kernel, I visualised posterior means and uncertainties for all models to check for pathological behaviour.

Exploration vs exploitation is handled using UCB:

where the uncertainty term encourages exploration early on.

Model Refinement

As more data became available, I placed increasing emphasis on kernel diagnostics (length scales and noise levels), sanity-checking 1D conditional slices, and identifying when models were underdetermined or overconfident. In some cases, I applied output transformations (e.g. log-transforming outputs) when heavy-tailed behaviour was observed.

I also introduced repulsion terms in the acquisition function to discourage repeatedly sampling near previously queried points and encourage greater exploration, while recognising that in higher dimensions Euclidean distance becomes less informative.

Low Data Regimes

For very small sample sizes, I found that BO could produce misleading structure. In these cases, I explored simpler models such as additive linear models and response surfaces to prevent overfitting when the data do not support complex structure.

For Function 1, the output was almost entirely near-zero values. I disregarded this as noise and prioritised pure exploration to identify the areas with the lowest sampling density.

Summary

I adopted a thoughtful approach by adapting model complexity to data availability and used visual diagnostics extensively to validate modelling assumptions.
