## Function 7 Overview

Six-dimensional data representing the hyperparameters of a ML model with an output of model performance score.

## Nature of the data

Input: (30,6) array

Output: (30,) array



The various features have differing impact. Some, like dimensions 1, 4, and 5 have a strong effect on the output, while others are weaker. However, not all kernels agree: the Matérn + RBF kernel infers a strong effect from Dimension 6 while Matérn alone infers a completely flat reponse.



The outputs are strictly positive and vary over several orders of magnitude. However, several evaluations at different points have yielded near-identical results of 2.92-2.94, indicating there may be a limit on the performance of the model.



Estimates of SNR in decibels ranged from 8 to 21, considered poor to fair. This indicates noise is relatively high in this problem.

## Your optimisation strategy

I used Gaussian Processes throughout to choose the next points. began using UCB with a high kappa that I gradually reduced, later switching to EI to exploit the identified peak. I focussed mainly on the RationalQuadratic kernel as this fitted a smoother curve to the data that predicted unimodal peaks, whereas most other kernels early on predicted linear responses, with suggested points at the extremes. As time went on, RationalQuadratic became a less good fit and the rougher kernels started to pick out unimodal peaks, so I switched over,

## Data handling and preprocessing

Due to the strictly positive output and variance over multiple orders of magnitude I used a log transform prior to using the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

After initially good progress in the first few weeks, progress became quite variable. Four separate guesses yielded results of 2.92-2.94, indicating an upper limit may have been reached. Subsequent moves away from this region did not result in further improvement

## Performance and results

The best output was 2.94 at a vector of \[0.27552, 0.28032, 0.43690, 0.13466, 0.27493, 0.64890]. I am confident this value is close to the global maximum valuegiven four separate evaluations at nearly exactly this point. This is reflected in my strong performance in the leaderboard of 3rd.



The data seem to indicate some sets of hyperparameters generate close to random outputs, hence near-zero scores and drastic jumps when the correct combination is found. However, improvements cannot continue exponentially forever, so it seems there is a practical limit on performance and a general region that achieves similar results.

## Ethical, practical and general considerations

Bayesian Optimisation is a classic approach to hyperparameter tuning in the real-world



The synthetic nature of the function is a good approximation for a real-world benchmark model or large data.



My strategy would work well in a real-life environment as it is a tried and tested method and the impact of exploration having poorer results is low.

