## Function 5 Overview

Four dimensional data representing the concentrations of inputs to a chemical manufacturing process with an output of the yield. The function is said to be unimodal

## Nature of the data

Input: (20,4) array

Output: (20,) array



The yield is a strictly positive output ranging across several orders of magnitude. Dimensions 3 and 4 appear to have the strongest impact, with the highest yields seen at close to maximum concentrations, while in my analysis dimensions 1 and 2 have greater uncertainty, sloping upwards towards higher concentrations or peaking in the centre



Estimates of SNR in decibels ranged from 22 to 44, considered good to excellent. This indicates noise is moderate or low in this problem.

## Your optimisation strategy

I struggled with this problem. I used Gaussian Optimisation, but the uncertainty led to constant evaluations of extreme points (e.g., \[1.0, 1.0, 1.0, 1.0], or varying only one feature). I used UCB followed by EI in line with my approach to other problems, but neither approach seemed to yield different outcomes. A longer-term view that sought to fill out the gaps (perhaps by minimising entropy) rather than seeking to achieve the best res

## Data handling and preprocessing

I applied a log transformation to the data due to the strictly positive data and multiple orders of magnitude prior to using the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

I made rapid initial progress in this functions with steady improvements up to week seven as suggested inputs gradually converged on guesses close to \[1.0, 1.0, 1.0, 1.0]. Moving away from this peak yielded consistently inferior results and uncertainty remained high with even small movements of one feature. 

## Performance and results

The best output was 6073 at a vector of \[0.80539, 1.00000, 1.00000, 1.00000]. While it seems clear most of the inputs are relatively close to 1, I have not identified a clear peak. The exponential distribution of the data and the remaining uncertainty means there remains potential for very significant improvement. This is reflected in my relatively weaker performance in the leaderboard of 14th.



The data were described as unimodal so it is somewhat surprising I was unable to find this peak.

## Ethical, practical and general considerations

A real-world experiment would likely be able to take advantage of significantly more data than was available here, taking learnings from small-scale experiments using potentially hundreds of conditions to inform larger scale experiments. However, the general approach of using BO to solve a problem like this is a classic use-case.



The synthetic nature of the function means domain knowledge on the interaction between the various components could not be used.



My strategy was not fully successful in capturing the variance of the data so it would need some refinement before deployed to at-scale manufacturing.

