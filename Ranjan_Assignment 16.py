# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:01:15 2018

@author: Ranjan
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
dta = sm.datasets.fair.load_pandas().data
# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
religious + educ + C(occupation) + C(occupation_husb)',
dta, return_type="dataframe")
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})
y = np.ravel(y)

print(X.describe())

logreg=LogisticRegression()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=99)

logreg.fit(X_train,y_train)

y_pred=logreg.predict_proba(X_test)

#printing training accuracy score
print("Training Score is: ")
print(logreg.score(X_train,y_train))

#printing test accuracy score
print("\nTest Score is: ")
print(logreg.score(X_test,y_test))

# plotting test label data and predicted data
plt.subplot(2,1,1)
plt.plot(y_test)
plt.subplot(2,1,2)
plt.plot(y_pred)



