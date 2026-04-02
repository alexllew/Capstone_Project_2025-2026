## Function 3 Overview

Three dimensional data representing the amounts of three compounds used in a drug discovery project. The outputs represent the negative of side-effects. In other words, the higher the value the lower the negative impact of the combination.

## Nature of the data

Input: (15,3) array

Output: (15,) array



Dimension 1 is consistently predicted to have minimal impact on side effects, while Dimension 2 has a moderate impact. There is some uncertainty here as some kernels predict a higher score in the middle of the distribution, and others closer to 1. However, the change is not large across the full range. 



Dimension 3 has the most prominent effect, with a large outlier of major side-effects seen at the highest concentration, in between, two-three peaks are observed: one broader peak in the centre, and two smaller peaks towards the edge. Each of these has a similar score, although I currently have greater uncertainty around the smaller peak closer to 1.0 as different kernels disagree on whether it is real.



Three kernels yielded a Signal-to-Noise ratio in decibels between 20-26, which is considered good, while the Matérn + RBF kernel was an outlier with an excellent SNR of 36. This indicates this is a relatively low-noise environment.

## Your optimisation strategy

I used Gaussian Processes throughout to choose the next points. Similar to function 2, I initially believed kernels were overfitting to dimension 3 and I imposed length scale restrictions to prevent dimension 2 being completely ignored, later switching to using RationalQuadratic which fitted a smoother curve. However, towards the end of the process it became clear that this was not overfitting, and I moved towards trusting the models that predicted shorter length scales for dimension 1.



I began using UCB with a high kappa that I gradually reduced, later switching to EI to exploit the identified peak.

&#x20;

## Data handling and preprocessing

I used the StandardScaler to ensure the outputs were centred on zero with unit variance but did not perform any other data pre-processing.

## Weekly iteration and learning

I explored broadly across the range for both dimension 1 and 2, confirming they have relatively little impact on the output. My evaluations for dimension 3 also focussed on exploration, but avoided some areas, such as 0.1-0.3 as this already had excellent coverage, and 0.9-1.0 as it has been seen that this results in very negative outcomes. I wasted some guesses evaluating 0.0 as kernels were projecting significant uncertainty here but I mainly explored the central peak, leaving a gap in the potential small peak closer to 0.8.

## Performance and results

The best output was -0.007 at a vector of \[0.40888, 0.52811, 0.48658]. I have moderate confidence this is close to the global optimum but there remains a possibility that x3 closer to 0.8 would be the best result. Given my relatively worse performance of 17th in the leaderboard, there is considerable room for improvement.



It would be typically expect that side-effects would scale linearly, or have at most one peak, so the multimodal distribution observed is relatively surprising. However, this reflects the complex biological reality of such problems. It is to be expected that one dimension should dominate here as it is unlikely for all three compounds to be toxic unless there is some synergistic effect.

## Ethical, practical and general considerations

A real-world problem involving side-effects in patients would not normally be carried out this way due to the ethical problems associated with erratically increasing or decreasing the amounts of multiple compounds. Instead, a dose-escalation study of one, or at most two, compounds would be performed with healthy volunteers to identify the maximum tolerated dose.



The synthetic nature of the function contrasts with a real-world scenario in which biological data would be available from previous studies or similar compounds to put the data into perspective and contextualise decisions around dosing.



My strategy would not be appropriate to application to patients as the consequences of a mistake are severe. Instead, a much more cautious approach should be adopted where small increases or decreases are made in a large number of patients and experiments halted once a threshold of toxicity or side-effects is reached.



A future user should be aware that the data do not necessarily reflect a realistic scenario for real-world application.

