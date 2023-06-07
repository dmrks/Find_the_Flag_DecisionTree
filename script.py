import codecademylib3
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
#print(df.landmass.value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df.landmass.isin([6,3])]
print(df_36)

#Print the average vales of the predictors for Europe and Oceania
print(df_36.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels = (df["landmass"].isin([3,6]))*1

#Print the variable types for the predictors
#print(df_36[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies((df[var]))
print(data)

#Split data into a train and test set
x_train, x_test, y_train, y_test =train_test_split(data, labels, random_state = 1, test_size=0.4)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []

for i in depths:
    dt = DecisionTreeClassifier(random_state = 10, max_depth = i)
    dt.fit(x_train ,y_train)
    acc_depth.append(dt.score(x_test, y_test))

#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


#Find the largest accuracy and the depth this occurs = 0.8333333333333334 / 3
print(np.max(acc_depth))
best_depth = depths[np.argmax(acc_depth)]
print(best_depth)

#Refit decision tree model with the highest accuracy and plot the decision tree
plt.figure(figsize=(14,8))
dt = DecisionTreeClassifier(random_state = 10, max_depth = best_depth)
dt.fit(x_train ,y_train)
tree.plot_tree(dt, feature_names = x_train.columns,  
               class_names = ['Europe', 'Oceania'],
                filled=True)
plt.show()
plt.clf()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list

ccp = np.logspace(-3, 0, num=20)
acc_pruned = []

for i in ccp:
    dt = DecisionTreeClassifier(random_state = 1,max_depth = best_depth, ccp_alpha = i)
    dt.fit(x_train ,y_train)
    acc_pruned.append(dt.score(x_test, y_test))

#Plot the accuracy vs ccp_alpha
plt.plot(ccp, acc_pruned)
plt.xlabel('ccp_alpha')
plt.ylabel('acc_pruned')
plt.show()

#Find the largest accuracy and the ccp value this occurs = 0.8076923076923077 // best_acc = 0.001

print(np.max(acc_pruned))
best_acc = ccp[np.argmax(acc_pruned)]
print(best_acc)


#Fit a decision tree model with the values for max_depth and ccp_alpha found above m + #Plot the final decision tree

plt.figure(figsize=(14,8))
dt = DecisionTreeClassifier(random_state = 10, max_depth = best_depth,ccp_alpha = best_acc)
dt.fit(x_train ,y_train)
tree.plot_tree(dt, feature_names = x_train.columns,  
               class_names = ['Europe', 'Oceania'],
                filled=True)
plt.show()
plt.clf()
