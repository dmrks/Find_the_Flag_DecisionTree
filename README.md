# Find_the_Flag_DecisionTree

Find the flag!
Can you guess which continent this flag comes from?

Flag of Reunion
What are some of the features that would clue you in? Maybe some of the colors are good indicators. 
The presence or absence of certain shapes could give you a hint. In this project, we’ll use decision trees to try to predict the continent of flags based on several of these features.

We’ll explore which features are the best to use and the best way to create your decision tree.

Datasets
The original data set is available at the UCI Machine Learning Repository:

https://archive.ics.uci.edu/ml/datasets/Flags

# Investigate the data
1.
The dataset has been loaded for you in script.py and saved as a dataframe named df. Some of the input and output features of interest are:

name: Name of the country concerned
landmass: 1=N.America, 2=S.America, 3=Europe, 4=Africa, 5=Asia, 6=Oceania
bars: Number of vertical bars in the flag
stripes: Number of horizontal stripes in the flag
colours: Number of different colours in the flag
red: 0 if red absent, 1 if red present in the flag
…

mainhue: predominant colour in the flag (tie-breaks decided by taking the topmost hue, if that fails then the most central hue, and if that fails the leftmost hue)
circles: Number of circles in the flag
crosses: Number of (upright) crosses
saltires: Number of diagonal crosses
quarters: Number of quartered sections
sunstars: Number of sun or star symbols
We will build a decision tree classifier to predict what continent a particular flag comes from. Before that, we want to understand the distribution of flags by continent. Calcluate the count of flags by landmass value.


2.
Rather than looking at all six continents, we will focus on just two, Europe and Oceania. Create a new dataframe with only flags from Europe and Oceania.


3.
Given the list of predictors in the list var, print the average values of each for these two continents. Note which predictors have very different averages.



4.
We will build a classifier to distinguish flags for these two continents – but first, inspect the variable types for each of the predictors.



5.
Note that all the predictor variables are numeric except for mainhue. Transform the dataset of predictor variables to dummy variables and save this in a new dataframe called data.


6.
Split the data into a train and test set.



# Tune Decision Tree Classifiers by Depth

7.
We will explore tuning the decision tree model by testing the performance over a range of max_depth values. Fit a decision tree classifier for max_depth values from 1-20. Save the accuracy score in for each depth in the list acc_depth.


8.
Plot the accuracy of the decision tree models versus the max_depth.



9.
Find the largest accuracy and the depth this occurs.



10.
Refit the decision tree model using the max_depth from above; plot the decision tree.



# Tune Decision Tree Classifiers by Pruning

11.
Like we did with max_depth, we will now tune the tree by using the hyperparameter ccp_alpha, which is a pruning parameter. Fit a decision tree classifier for each value in ccp. Save the accuracy score in the list acc_pruned.



12.
Plot the accuracy of the decision tree models versus the ccp_alpha.


13.
Find the largest accuracy and the ccp_alpha value this occurs.



14.
Fit a decision tree model with the values for max_depth and ccp_alpha found above. Plot the final decision tree.
