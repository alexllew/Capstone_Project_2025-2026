# Overview
## Model Name
Dual-Engine Adaptive Bayesian Optimizer (DE-ABO)
## Model Type
Surrogate-based Black-Box Optimiser featuring a Gaussian Process (GP) multi-kernel ranking system and an Optuna-tuned feedforward Neural Network (NN) ensemble.

# Intended Use
The model is designed to optimising expensive-to-evaluate, continuous black-box functions. 
## Use-cases
Hyperparameter tuning for machine-learning models and other tasks requiring a balance of exploration and exploitation in low-to-moderate dimensional spaces. 
## Use-cases to avoid
High-dimensional combinatorial or discrete optimisation (L-BFGS-B acquisition optimisers are designed for continuous bounds), and extremely noisy environments with ultra-cheap evaluations, in which random search or purely evolutionary algorithms may be more  computationally efficient.

# Details
The framework can seamlessly switch between different acquisition functions and surrogate models across optimisation rounds. For an optimisation campaign with limited evaluations, the system employs the following progressive strategies:

## Early Rounds (Exploration Phase)
The framework relies on the Upper Confidence Bound (UCB) acquisition strategy with a standard κ (e.g., 1.96) or Expected Improvement (EI). The GP engine evaluates a diverse bank of kernels (Matern 2.5, RationalQuadratic, Linear + Matern, and an RBF + Matern mixture), using cross-validation to find the topology of the unknown function.

## Middle Rounds (Adaptive Phase)
As historical data builds in history.csv, the framework can pivot based on landscape complexity. If the GP struggles with highly non-linear or discontinuous spaces, the system utilizes the NeuralBayesianOptimizer. This uses optuna to perform Neural Architecture Search (NAS) to build an ensemble of PyTorch models, capturing complex global trends better than a stationary GP kernel.
 
## Late Rounds (Exploitation & Fine-Tuning)
The framework exploits the surrogate surface using heavy local optimisation. For the GP, it uses a multi-restart L-BFGS-B optimizer seeded by a Latin Hypercube Sample (qmc.LatinHypercube). For the NN ensemble, it uses Differential Evolution (scipy.optimize.differential_evolution). A repulsion penalty is dynamically applied to the acquisition score to prevent the optimiser from repeatedly querying identical coordinates in flat optimal regions.

# Performance

## GP Selection Metrics
Models are ranked using Cross-Validation Mean Squared Error (neg_mean_squared_error) and Log Marginal Likelihood to ensure the surrogate accurately maps the underlying function before suggesting the next point.

## NN Ensemble Metrics
The neural network relies on a 5-fold CV MSE to evaluate architectures during the Optuna trials, ensuring the ensemble generalises well to unseen regions of the parameter space.

## Acquisition Score
Tracks the magnitude of predicted improvement, transformed back to the original target scale for accurate display and logging.

Results for the eight functions are summarised below:
## Function 1
The model has struggled to identify structure in this function, with nearly all evaluations resulting in near-zero results. 
## Function 2
Most kernels have fitted long-length scales to feature 2, resulting in a complex multi-peaked fit to feature 1, with visual evidence of overfitting when plotted. RationalQuadratic has a smoother fit for both features, identifying a clear peaked region in both. The Matern kernel has consistently produced results somewhere between the two. Each has been used to balance exploration with exploitation, and ensure no single kernel is excessively relied upon
## Function 3
This function has proven to be relatively flat, without major peaks and troughs. Similarly to Function 2, most kernels identify little or no correlation with Feature 1, and a simple linear correlation with Feature 2, with Feature 2 most prioritised. RationalQuadratic again fits a smoother curve, with a sinusoisal relationship seen in all three features. Point selection has balanced improving the RationalQuadratic fit with reducing the uncertainty seen in regions of Feature 3 in the other kernels.
## Function 4
This function was rapidly identified to have a single monotonic peak close to the centre of each feature, with close agreement between different kernels. Optimising this function has focussed on marginal improvements through exploitation.
## Function 5
This function proved to have a logarithmic distribution and initially very rapid progress was made. However, the limitations of larger search spaces began to show themselves as the model began to predict extremes (values of 1.00000), varying only one or two features. While this enabled some high values to be achieved, this also limited exploration of some areas of the search space. A pivot to using the Neural Network identified some alternative regions to explore.
## Function 6
There was general agreement between kernels on the overall shape of this function, with differences in the importance of each feature. The shape appears to be characterise by a single optimal region for each feature, with optimisation focussing on improving understanding of the location of this region as other features vary.
## Function 7
Like function 5 this proved to have a logarithmic distribution and rapid initial progress was made. Interestingly, four separate evaluations have resulted in very similar values of 2.92-2.93, which suggests there may be some sort of hard-cap to this function.
## Function 8
This function reached a plateau relatively rapidly. The large search space meant the model had a tendency to shoot towards extreme values and initial attempts were made to offset this with a repulsion mechanism. Ultimately it proved more fruitful to evaluate these extreme points to eliminate the uncertainty in these regions.
# Assumptions and Limitations
## Target Scaling Assumptions
The code relies heavily on StandardScaler and log transformations. It assumes that extreme variations in target outputs can be normalised effectively. If a log transform is forced on non-positive values, it will fail or distort the optimisation surface.
## Maximisation
The acquisition functions calculate improvements relative to current_best_y = self.y_norm.max(). The framework assumes an objective of maximisation (or that minimisation tasks have had their targets inverted prior to ingestion).
## Computational Bottleneck
The standard GP scales at O(N3) regarding the number of data points. The secondary NN engine mitigates this on larger datasets, but training an ensemble of NNs via Optuna per optimization step introduces high computational overhead.
# Ethical Considerations
## Transparency and Explainability
The framework combats the black-box nature of both the objective function and the NN/GP surrogates through its robust Plotter class. By plotting 1D slice views of the predicted mean, the 1.96 standard deviation confidence bounds, and the overlaying acquisition surface, practitioners can visually verify why the algorithm chose a specific coordinate.
## Reproducibility
A central random seed is propagated across Numpy, PyTorch, SciPy's QMC samplers, and Optuna. This strict state management ensures that identical initial conditions (inputs.npy, outputs.npy) will yield the exact same sequence of suggested coordinates, which is critical for auditing algorithmic decisions in real-world engineering or scientific applications.
