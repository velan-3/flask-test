# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:41:36 2023

@author: Velan
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

#importing dataset
data = pd.read_csv("insurance.csv")
data

a = data.iloc[:,0:4]
a

def convert_to_int(word):
    word_dict = {'male':1,'female':0}
    return word_dict[word]

a['sex'] = a['sex'].apply(lambda x: convert_to_int(x))

def convert_to_intt(word):
    word_dict = {'yes':1,'no':0}
    return word_dict[word]
a['smoker'] = a['smoker'].apply(lambda x: convert_to_intt(x))

#def convert_to_inttt(word):
    #word_dict = {'northeast':1,'northwest':0,'southeast':2,'southwest':3}
    #return word_dict[word]
#a['region'] = a['region'].apply(lambda x: convert_to_inttt(x))


y = data.iloc[:,-1]
y

X_train, X_test,Y_train, Y_test = train_test_split(a,y,test_size=0.2,random_state=22)
X_train.shape, X_test.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = RandomForestRegressor()
models.fit(X_train,Y_train)
models.predict(X_test)

pickle.dump(models ,open("model.pkl","wb"))
model = pickle.load(open("model.pkl","rb"))