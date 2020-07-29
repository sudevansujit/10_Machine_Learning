# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:47:13 2020

@author: Sujit

Feature Selection Techniques
"""

import pandas as pd
import numpy as np

data = pd.read_csv('Data/mobile_dataset.csv')
data.head()

data.shape

#Splitting Data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X.head()
y.head()

# Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Applying SelectKBest algorithm
ordered_rank_features = SelectKBest(score_func = chi2, k = 10)
ordered_features = ordered_rank_features.fit(X, y)
ordered_features.scores_

scores = pd.DataFrame(np.round(ordered_features.scores_, 3), index = X.columns)
scores

scores = scores.reset_index()
#scores.columns.map({'index':'Columns', '0': 'Score'} )
scores.columns = ['Columns', 'Score']
scores

# Finding 5 largest
scores.nlargest(10, 'Score')

# Feature Importance Technique
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

etc = ExtraTreesClassifier()
etc.fit(X, y)

ranked_features = pd.Series(etc.feature_importances_, 
                            index = X.columns)

# Viewing in a plot
ranked_features.nlargest(10).plot(kind = 'barh')

# Function to find highly correlated features
def high_corr(dframe, thresh = 0.5):
    df = dframe.corr()
    corr_list = []
    for i in df.index:
        for j in df.columns:
            if ((i != j) & (df.loc[i, j] > 0.5)) | ((i != j) & (df.loc[i, j] < -0.5)) :
                res = (i, j, df.loc[i, j])
                corr_list.append(res)
    return corr_list

high_corr(X)


import seaborn as sns
sns.heatmap(X.corr()[(X.corr() > 0.5) | (X.corr() < -0.5)], annot = True)


# Information Gain
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X, y)
mutual_data = pd.Series(mutual_info, index = X.columns)
mutual_data.sort_values(ascending = False)














