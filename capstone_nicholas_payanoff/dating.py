#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:10:35 2018

@author: Nicholas Payanoff
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Create your df here:

data = pd.read_csv('profiles.csv')

"""There are multiple options for each of the categories. For example, 
Sign has more than 12 options because it also takes into account how serious
the person is about it. This section is going to clean that up a bit,
extracting just the sign/religion/etc from the option the user chose

This is also where mapping features to numbers will occur
"""
#defining the maps
job_switch = {'other':0, 'student':1, 'science':2, 'computer':3, 'artistic':4,
              'sales':5, 'medicine':6, 'education':7, 'executive':8, 
              'banking':9, 'entertainment':10, 'law':11, 'hospitality':12,
              'construction':13, 'clerical':14, 'political':15, 'rather':16,
              'transportation':17, 'unemployed':18, 'retired':19, 'military':20}
ethnicity_switch = {'other':0, 'white':1, 'black':2, 'asian':3, 'middle':4,
                    'pacific':5, 'indian':6, 'native':7, 'hispanic':8}
signs_map = {'leo':0, 'libra':1, 'cancer':2, 'virgo':3, 'scorpio':4, 
         'gemini':5, 'taurus':6, 'aries':7, 'pisces':8, 
         'aquarius':9, 'sagittarius':10, 'capricorn':11}
drink_mapping = {'not at all': 0, 'rarely':1, 'socially':2, 'very often':3, 
                 'desperately':4}
smoke_map = {'no':0, 'trying to quit':1, 'when drinking':2, 'sometimes':3,
             'yes':4}

religion_map = {'agnosticism':0, 'other':1, 'atheism':2, 'christianity':3,
                'catholicism':4, 'judaism':5, 'buddhism':6, 'hinduism':7,
                'islam':8}
drugs_map = {'never':0, 'sometimes':1, 'often':2}

#creating columns from maps
data['signs_trimmed'] = data.sign.apply(lambda x: str(x).split()[0])
data['religion_trimmed'] = data.religion.apply(lambda x: str(x).split()[0])
"""
religion and sign are getting 2 columns so that they may be used as labels
"""
data['job_code'] = data.job.apply(lambda x: job_switch.get(str(x).split()[0]))
#for ethinicity, I will assume that the first listed is the primary one
data['ethnicity_code'] = data.ethnicity.apply(lambda x: 
    ethnicity_switch.get(str(x).split()[0]))
data['num_speaks'] = data.speaks.apply(lambda x: len(str(x).split(',')))
data['signs_code'] = data.signs_trimmed.map(signs_map)
data['drink_code'] = data.drinks.map(drink_mapping)
data['smoke_code'] = data.smokes.map(smoke_map)
data['drugs_code'] = data.drugs.map(drugs_map)
data['religion_code'] = data.religion_trimmed.map(religion_map)
#"borrowing" the code from the instructions for getting the essays
essay_cols = ['essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5',
              'essay6', 'essay7', 'essay8', 'essay9']
all_essays = data[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
data['essays_len'] = all_essays.apply(lambda x: len(x))

"""filling some NA values with 'other'. since they weren't specified, 'other'
should be a safe guess"""

data.fillna({'ethnicity_code':0, 'job_code':0}, inplace = True)
 

  
#%% plotting some of the data
plt.hist(data['signs_trimmed'], bins=25, rwidth=0.80, align='mid')
plt.xlabel('Sign')
plt.xticks(rotation=270)
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Sign.png')
plt.close()
plt.hist(data['sex'], bins=3)
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Sex.png')
plt.close()
plt.hist(data['religion_trimmed'], bins=24, align='mid')
plt.xlabel('Religion')
plt.xticks(rotation=270)
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Religion.png')
plt.close()
#%% Naive Bayes - determine sex/ethnicity by words used
counter = CountVectorizer()
counter.fit(all_essays)

NBtraining_data, NBvalidation_data, NBtraining_labels, NBvalidation_labels = \
train_test_split(all_essays, data['sex'], random_state=13)

NBtrain_counts = counter.transform(NBtraining_data)
NBvalidation_counts = counter.transform(NBvalidation_data)

NBclassifier = MultinomialNB()
NBclassifier.fit(NBtrain_counts, NBtraining_labels)
NBpredict = NBclassifier.predict(NBvalidation_counts)
#scores
print('Naive Bayes classifier: Sex')
print('Score: %f' %(NBclassifier.score(NBvalidation_counts,NBvalidation_labels)))
print('Accuracy: %f' %(accuracy_score(NBvalidation_labels, NBpredict)))
print('Recall: %f' %(recall_score(NBvalidation_labels, NBpredict,
                                   average='macro')))
print('Precision: %f' %(precision_score(NBvalidation_labels, NBpredict, 
                                       average='macro')))
print('F1: %f \n' %(f1_score(NBvalidation_labels, NBpredict,
                          average='macro')))
#using NB to classify ethnicity from essays
NBtraining_data, NBvalidation_data, NBtraining_labels, NBvalidation_labels = \
train_test_split(all_essays, data['ethnicity_code'], random_state=13)

NBtrain_counts = counter.transform(NBtraining_data)
NBvalidation_counts = counter.transform(NBvalidation_data)

NBclassifier.fit(NBtrain_counts, NBtraining_labels)
NBpredict = NBclassifier.predict(NBvalidation_counts)
#scores
print('Naive Bayes classifier accuracy: Ethnicity')
print('Score: %f' %(NBclassifier.score(NBvalidation_counts,NBvalidation_labels)))
print('Accuracy: %f' %(accuracy_score(NBvalidation_labels, NBpredict)))
print('Recall: %f' %(recall_score(NBvalidation_labels, NBpredict,
                                   average='macro')))
print('Precision: %f' %(precision_score(NBvalidation_labels, NBpredict, 
                                       average='macro')))
print('F1: %f \n' %(f1_score(NBvalidation_labels, NBpredict,
                          average='macro')))
#%% Regression - predict income from features

LRfeatures = ['smoke_code', 'drink_code', 'drugs_code', 'signs_code',
              'religion_code', 'essays_len', 'age', 'job_code', 'num_speaks']
# removing -1 income and nan features
LRdata = data[data.income != -1]
LRdata.dropna(subset=LRfeatures, inplace=True)
LRxdata = LRdata[LRfeatures]
#normalizing
min_max_scaler = preprocessing.MinMaxScaler()
LRxdatascaled = min_max_scaler.fit_transform(LRxdata)
#linear regression model
LRxtrain, LRxtest, LRytrain, LRytest = \
train_test_split(LRxdatascaled, LRdata['income'], random_state=13)
LRmodel = LinearRegression()
LRmodel.fit(LRxtrain, LRytrain)
LRpredict = LRmodel.predict(LRxtest)
#scores
print('Linear Regression Model')
print('Score: %f \n' %(LRmodel.score(LRxtest, LRytest)))

# K regressor model
k_test_score = []
for i in range(1,10):
    KRmodel = KNeighborsRegressor(n_neighbors=i)
    KRmodel.fit(LRxtrain, LRytrain)
    k_test_score.append(KRmodel.score(LRxtest, LRytest))
#scores
print('KNeighbors Regressor')
print('Score for k=5: %f \n' %(k_test_score[4]))
#plot accuracy vs k
plt.plot(range(1,10), k_test_score)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('K Neighbors Regressor')
plt.savefig('kneighborsregressor.png')
plt.close()
#%%classify ethnicity from behavior - Kneighbors

Kfeatures = ['smoke_code', 'drink_code', 'drugs_code', 'religion_code', 
                 'age', 'num_speaks', 'ethnicity_code']
Kdata = data[Kfeatures]
Kdata.dropna(inplace = True)
Kxdata = Kdata[['smoke_code', 'drink_code', 'drugs_code', 'religion_code', 
                 'age', 'num_speaks']]
#normalizing
min_max_scaler = preprocessing.MinMaxScaler()
Kdatascaled = min_max_scaler.fit_transform(Kxdata)
k_training_data, k_validation_data, k_training_labels, k_validation_labels = \
train_test_split(Kdatascaled, Kdata['ethnicity_code'], random_state = 13)
accuracies = []
for i in range(1,10):
    Kmodel = KNeighborsClassifier(n_neighbors = i)
    Kmodel.fit(k_training_data, k_training_labels)
    if i == 5:
        #print scores for k = 5
        kpredict = Kmodel.predict(k_validation_data)
        print('K Neighbors Classifier')
        print('Score: %f' %(Kmodel.score(k_validation_data, k_validation_labels)))
        print('Accuracy: %f' %(accuracy_score(k_validation_labels, kpredict)))
        print('Precision: %f' %(precision_score(k_validation_labels, kpredict, average='macro')))
        print('Recall: %f' %(recall_score(k_validation_labels, kpredict, average='macro')))
        print('F1 Score: %f' %(f1_score(k_validation_labels, kpredict, average='macro')))
        print('')
    accuracies.append(Kmodel.score(k_validation_data, k_validation_labels))
#plot accuracy vs k
plt.plot(range(1,10), accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('K Neighbors Classifier')
plt.savefig('kneighnorsclassifier.png')
plt.close()

    







#%%playing with plots to get it to look nice
"""
plt.hist(data['signs_trimmed'], bins=25, rwidth=0.80, align='mid')
plt.xlabel('Religion')
plt.xticks(rotation=270)
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
"""




    

