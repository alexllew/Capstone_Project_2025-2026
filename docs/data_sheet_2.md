## Function 2 Overview

Two dimensional data representing a generic black box or ML model that returns a log-likelihood score. The data is noisy and has local maxima that do not represent the global maximum

## Nature of the data

Input: (10,2) array

Output: (10,) array



From the beginning, most kernels consistently identified dimension 1 to be the most important, with little or no influence from dimension 2. Many local optima are apparent but one peak was notably higher. Guesses ranged across the feature space for dimension 2, but focussed around this main peak for dimension 1. This confirmed dimension 2 had little influence as similar values were obtained with radically different inputs for this feature. Guesses close to 0 and 1 in dimension 1 were also made to confirm no peaks existed at the edge of the feature space



Three kernels yielded a Signal-to-Noise ratio in decibels between 12-20, which is considered fair, while RationalQuadratic was an outlier with a very high SNR of 40. This indicates while there is some noise in the system, this is a relatively minor component.

## Your optimisation strategy

I used Gaussian Processes throughout to choose the next points. Initially, I thought there may be some overfitting, with kernels running directly through all the points in dimension 1. I imposed length scale restrictions to prevent dimension 2 being completely ignored, and later switched to using RationalQuadratic which fitted a smoother curve. However, towards the end of the process it became clear that this was not overfitting, and I moved towards trusting the models that predicted shorter length scales for dimension 1.



I began using UCB with a high kappa that I gradually reduced, later switching to EI to exploit the identified peak.

&#x20;

## Data handling and preprocessing

I used the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

As I explored widely across dimension 2 it became clear it had little influence on the outcome and I began to trust the output of kernels that placed more emphasis on dimension 1. Exploration in dimension 1 was restricted to the major peak identified as these showed significantly higher values than other regions. Given that dimension 2 did not impact the output much it seemed unlikely the other peaks were likely to be increased by enough to be competitive. I did evaluate extreme points as these had much higher uncertainty but both returned poor result.

## Performance and results

The best output was 0.73 at a vector of \[0.696202, 0.889319]. While there is room for refinement, I believe this is close to the global optimum, though there is a low probability that a peak at around x1 = 0.24 could be higher. I finished 10th overall in this function which suggests that while it was not a perfect outcome, there is likely not a huge scope for improvement.



The results align with expectations for this function given the prior information that there are multiple local optima, although it is a little surprising that it seems to break down to a near-one-dimensional problem

## Ethical, practical and general considerations

There is real-world application to black-box function where the underlying structure is unknown and dimensions may be relevant or non-relevant with no prior information. It has provided value in appreciating that we do not always need to treat all inputs as relevant just because they are there.



The synthetic nature of the function simulates exactly the problem described: a mystery ML model.



My strategy would likely scale to real-world problems but context is important to understand the cost of exploration and whether can take risks on low-probability chances (such as one of the minor peaks having a narrow region of very high performance)



A future user should be aware that it is important to avoid human bias in assuming a model is wrong just because it produces unexpected results, such as very long or short length scales for certain input features.

