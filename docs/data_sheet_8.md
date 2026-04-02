## Function 8 Overview

Eight-dimensional data representing the inputs of a black model with a generic performance/efficiency output. The data are complex and identifying local optima is a practical strategy.

## Nature of the data

Input: (40,8) array

Output: (40,) array



Some dimensions, such as 1, 3, and 7 have strong effects with clearly defined peaks, while others exhibit flatter or more linear trends.



Estimates of SNR in decibels ranged from 36 to 66, indicating extremely low noise.

## Your optimisation strategy

I used Gaussian Processes throughout to choose the next points. began using UCB with a high kappa that I gradually reduced, later switching to EI to exploit the identified peak. I focussed mainly on the RationalQuadratic kernel as this fitted a smoother curve to the data that predicted unimodal peaks, whereas most other kernels early on predicted linear responses, with suggested points at the extremes. 

## Data handling and preprocessing

I used the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

Modest but steady improvement was observed, with the best result obtained in the final week. Due to the high dimensionality of the data, and limited evaluations available, I chose to focus on exploiting the peaks already identified in week one, especially as new data would not significantly expand the dataset. 

## Performance and results

The best output was 9.98 at a vector of \[0.11372, 0.18460, 0.10511, 0.20328, 0.76949, 0.48404, 0.19825, 0.29558]. I have moderate confidence this is close to the global optimum, as all kernels now show a consistent structure to the data. However, this is a high-dimensional problem and it is entirely possible I have missed another peak. This is reflected in my strong performance in the leaderboard of 7th, indicating limited room for improvement.



## Ethical, practical and general considerations

Bayesian Optimisation is a classic approach to solving black-box optimisation problems



The synthetic nature of the function is precisely the kind of problem the data are supposed to simulate so it is suitable.



My strategy would work well in a real-life environment as it is a tried and tested method and the impact of exploration having poorer results is low.

