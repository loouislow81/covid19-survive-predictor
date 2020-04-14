#!/usr/bin/python3 

## 
# COVID-19 Disease Survival Predictor
# @file: predict.py
# @author: Loouis Low <loouis@gmail.com>
# @github: loouislow81/covid19-survive-predictor
# @license: MIT
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')


title = '\x1b[0;37;44m' + ' COVID-19 Disease Survival Predictor ' + '\x1b[0m' + '\n'
sys = '\x1b[0;30;47m' + ' sys ' + '\x1b[0m'
live = '\x1b[0;37;42m' + ' LIVE ' + '\x1b[0m'
dead = '\x1b[0;37;41m' + ' DEAD ' + '\x1b[0m'
print("\n" + title + "")


"""
  I only managed to find this COVID-19 dataset at Kaggle with death and age columns.
  (https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
"""
print(sys + " reading raw dataset from .csv")
covi19 = pd.read_csv('./dataset/COVID19_line_list_data.csv')
covi19.head()


print(sys + " processing the data and remove artifacts data")
covi19 = covi19[['gender', 'age', 'death']]
covi19 = covi19[covi19['gender'].notna()]
covi19 = covi19[covi19['age'].notna()]
covi19['gender'].replace(['male', 'female'], [0, 1], inplace=True)
covi19.tail()


"""
  some data are broken or incorrect and has to be removed
  before tossing into the model for training that would
  disrupting the result. Had to do it manually.
"""
print(sys + " cleaning bad data...")
covi19['death'].replace(['2/21/2020'], [1], inplace=True)
covi19['death'].replace(['2/21/2020'], [1], inplace=True)
covi19['death'].replace(['2/19/2020'], [1], inplace=True)
covi19['death'].replace(['2/19/2020'], [1], inplace=True)
covi19['death'].replace(['02/01/20'], [1], inplace=True)
covi19['death'].replace(['2/27/2020'], [1], inplace=True)
covi19['death'].replace(['2/25/2020'], [1], inplace=True)
covi19['death'].replace(['2/22/2020'], [1], inplace=True)
covi19['death'].replace(['2/24/2020'], [1], inplace=True)
covi19['death'].replace(['2/23/2020'], [1], inplace=True)
covi19['death'].replace(['2/26/2020'], [1], inplace=True)
covi19['death'].replace(['2/23/2020'], [1], inplace=True)
covi19['death'].replace(['2/23/2020'], [1], inplace=True)
covi19['death'].replace(['2/23/2020'], [1], inplace=True)
covi19['death'].replace(['2/25/2020'], [1], inplace=True)
covi19['death'].replace(['2/27/2020'], [1], inplace=True)
covi19['death'].replace(['2/26/2020'], [1], inplace=True)
covi19['death'].replace(['2/28/2020'], [1], inplace=True)
covi19['death'].replace(['2/13/2020'], [1], inplace=True)
covi19['death'].replace(['2/26/2020'], [1], inplace=True)
covi19['death'].replace(['2/14/2020'], [1], inplace=True)


"""
  creating new model based on sex, age and death
"""
print (sys + " creating model...")
model = KNeighborsClassifier()
y = covi19['death']
y=y.astype('int')
X = covi19.drop('death', axis=1)


print(sys + " training model...")
model.fit(X, y)
model.score(X, y)


def predict(model):

    # input
    gender = int(input('\nYour gender: (male or female) = (0 or 1): '))
    age = int(input('Your age: '))

    x = np.array([gender, age]).reshape(1, 2)
    predic = model.predict(x)
    test = model.predict_proba(x).T

    # supervise results
    survive = test.item(0)
    die = test.item(1)

    # normalize results
    survive_text = str(survive*100)
    die_text = str(die*100)

    print("\n" + sys + " output neurons: [" + str(survive) + ", " + str(die) + "]\n")

    # output
    if predic == 0:
        print(live + " you have " + survive_text + " % chance of surviving and\n       " + 
              die_text + "% chance of dying based on the dataset.\n")
    if predic == 1:
        print(dead + " you have " + survive_text + " % chance of surviving and\n       " + 
              die_text + "% chance of dying based on the dataset.\n")

predict(model)
