## Function 1 Overview

Data representing a two-dimensional area yielding non-zero readings only in proximity to potential contamination sources, such as in a radiation field. Ten data points were initially provided.

## Nature of the data

Input: (10,2) array

Output: (10,) array



The dataset yielded entirely near-zero values for the first ten evaluations, with the eleventh evaluation yielding a massively higher result of 1.96. This outcome was largely serendipitous as no underlying structure had been identified. No improvement on this score was possible in the remaining two weeks.



The data appear extremely sparse with at least one narrow peak against mostly low-level background noise

## Your optimisation strategy

Given the array of near-zero values, I opted for pure exploration. I selected the point which had the greatest Euclidean distance from the nearest previously evaluated point to ensure optimal coverage. When this method began to suggest extreme points, I switched to random search.



Once I identified a peak in week 11, I switched to Bayesian Optimisation with EI to see if it was possible to improve on this result. However, because so little data was available in this region, it was impossible to determine the width of this peak and subsequent evaluations yielded worse results.

## Data handling and preprocessing

Initially I did not apply any normalisation to these data as I was using pure exploration. Once the peak was identified I applied a symlog transformation, which is a method that allows a log transform to be applied to data with negative values. This was performed because the identified peak was orders of magnitude larger than other values in the dataset

## Weekly iteration and learning

1. For the most part new data points did not change my understanding of the function as they were all noise. The most informative point was the chance evaluation of a narrow local minimum. I am not sure what better option there was to have found this peak earlier.

## Performance and results

The best output was 1.96 at a vector of \[0.627301, 0.626080]. I have low confidence that this is near the global maximum as it is clear this dataset has very narrow peaks so there could easily be another significantly higher peak that was identified.



The results align with expectations for this function given that signal is only expected in proximity to the contamination sources.

## Ethical, practical and general considerations

This relates to real-world problems with sparse solutions in which most inputs simply results in background noise and only very specific combinations have any signal.



The synthetic nature of the function means there are no other sources of information to determine if there are more plausible regions to search for a peak and therefore the strategy had to rely on random search.



My strategy would not scale to a real-world scenario of searching for contamination as it only encountered one peak by chance and does not rule out the existence of other contamination sources.



A future user should be aware that the provided data likely do not contain useful information to identify a peak and caution should be applied before attempting to fit a model to it.

