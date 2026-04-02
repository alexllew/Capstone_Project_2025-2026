## Function 6 Overview

Five-dimensional data representing the quantity of ingredients in a cake recipe, with the output representing an assessment of cake quality by an expert taster

## Nature of the data

Input: (20,5) array

Output: (20,) array



All dimensions appear similarly impactful on the output, with Dimensions 1-4 having a clear unimodal peak, while dimension 5 may have a peak close to zero or reach a peak at zero itself.



Estimates of SNR in decibels ranged from 19 to 31, considered fair to excellent. This indicates noise is moderate or low in this problem.

## Your optimisation strategy

I used Gaussian Processes throughout to choose the next points. began using UCB with a high kappa that I gradually reduced, later switching to EI to exploit the identified peak. From the beginning, evaluation focussed on the exploiting the main peaks, with little exploration elsewhere as these had significantly worse predicted outcomes.

## Data handling and preprocessing

I used the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

After initially good progress in the first few weeks, progress stalled, indicating I had reached close to the global optimum and the aim from this point was on refinement of the exact location.

## Performance and results

The best output was -0.0777 at a vector of \[0.43137, 0.40864, 0.66077, 0.79248, 0.07306]. I am confident this is close to the global optimum as other regions have much lower predicted values with high confidence. This is reflected in my strong performance in the leaderboard of 3rd.



It is unsurprising that the data are unimodal given the nature of the problem: baking typically has a fairly rigid set of conditions that work well.

## Ethical, practical and general considerations

This approach would be applicable to real-world applications such as refining a recipe using customer feedback to optimise taste.



The synthetic nature of the function means that it does not capture the real-world variability of individual tastes and instead it is simulating an idealised case where quantitative feedback on a subjective question can be trusted.



My strategy would work well in a real-life environment as relatively few recipes could be tested to determine the optimal components for commercial deployment.

