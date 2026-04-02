## Function 4 Overview

Four dimensional data representing hyperparameters in a ML model used to approximate expensive calculations to determine optimal placing of products across warehouses for online sales. The output represents the difference between the output of the model and the expensive baseline. It is stated that the system is dynamic and full of local optima.

## Nature of the data

Input: (30,4) array

Output: (30,) array



All four dimensions have a meaningful impact with a single monotonic peak seen towards the centre of the feature space. This is a consistent pattern across kernels and was observed from the very beginning



Estimates of SNR in decibels ranged from 37 to 47. This indicates noise is low or minimal in this problem.

## Your optimisation strategy

I used Gaussian Processes throughout to choose the next points. It became clear early one that there was only a single peak in each dimension and the probability of exploration finding another was very low. Consequently, I switched to EI early and used smaller and smaller xi values as I approached the optimum.

## Data handling and preprocessing

I used the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

Nearly all my evaluations were focussed on the centre of the monotonic peaks, aimed at refining the exact position. As progress plateaued, I took one low-probability guess outside this region and the results were much lower as expected and in line with central estimates for the kernels in question. I therefore returned exclusively to exploitation.

## Performance and results

The best output was 0.72 at a vector of \[0.41765, 0.42155, 0.42024, 0.41122]. I have very high confidence this is close to the global optimum as all kernels have consistently predicted a single peak. I finished top of the leaderboard for this function, supporting this conclusion.



The outcome is unexpected given the description of the function as full of local optima. This turned out not to be the case at all.

## Ethical, practical and general considerations

A real-world simulation of logistics would most likely need to be constantly validated against the manual calculations as product offerings, warehouse availability, and customer-base change over time. If a fixed model were used it would likely drift over time and result in poor outcomes if not subject to ongoing process verification.



The synthetic nature of the function limits how applicable this problem is as complex human factors come into play: local regulations, rent, staffing availability, building maintenance and even factors like weather all play into business decisions on warehousing. Therefore a real-world application would need to use the ML model as a tool to inform decisions in a broader context rather than relying too strongly on the output.



My strategy would be broadly appropriate to the problem at scale as it can be deployed and tested without severe ramifications if it went wrong as there is always the manual backup calculation.



A future user should be aware that descriptions of data may not always be accurate (lack of local optima)

