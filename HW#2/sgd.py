#!/usr/bin/env python2
"""Stochastic Gradient Descent for Logistic Regression.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from math import exp
import random
import numpy as np
import pandas as pd

#Calculate logistic function
def logistic(x):
    r = float(exp(x))/float(1+exp(x)) 
    return r

#Calculate dot product of two lists
def dot(x, y):
    s = 0
    for i in range(len(x)):
        s = s + x[i]*y[i]
    return s

#Calculate prediction based on model
#w*x
def predict(model, point):
    x = point['features']
    w_x = dot(model,x)
    p = logistic(w_x)
    return p

#Calculate accuracy of predictions on data
# if p(Y=1|X,w)> 0.5 the label should be 1, then check label from data
# else, the label should be 0, then check label from data
def accuracy(data, predictions):
    correct = 0
    for i in range(len(data)):
        d = data[i]
        d_label = d["label"]
        if predictions[i]>= 0.5 and d_label == 1:
            correct += 1

        if predictions[i]< 0.5 and d_label == 0:
            correct += 1
    return float(correct)/len(data)

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

#Update model using learning rate and L2 regularization
#L2 norm is applied to trade off extreme large weights
#model is updated based on stochastic gradient ascent algorithm
def update(model, point, delta, rate, lam):
    update_delta = np.asarray(point)*delta
    regularization = -lam*np.asarray(model)
    gradient = regularization + update_delta
    model1 = model + rate*gradient
    return model1

#Train model using training data
#rate indicates learning rate, lam indicates penalty coefficient, epochs indicates iteration times
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    for j in range(epochs):
        for i in range(len(data)):
            x = data[i]
            point = x['features']
            label = x['label']
            delta = label - logistic(dot(model,point))
            model = update(model, point, delta, rate, lam)
    return model
# extract feature and label from csv file and store in to data list
def extract_features(raw):
    data = []
    processed_data1 = []
    processed_data2 = []
    # After trying to extract better feature, conslusion is that all the features contribe to the model
    feature_list = ["initial","age","type_employer","fnlwgt","education","education_num","marital","occupation","relationship","race","sex","capital_gain","capital_loss","hr_per_week","country"]
    numerical_list = ["age","fnlwgt","education_num","capital_gain","capital_loss","hr_per_week"]
    discrete_list = ["type_employer","education","marital","occupation","relationship","race","sex","country"]
    # preprocessing raw data
    for r in raw:
        point = {}
        point["label"] = (int(r['income'] == '>50K'))
        data.append(point)

        features_numerical = []
        features_discrete = []
        for i in range(1,len(feature_list)):
            if feature_list[i] in numerical_list:
                features_numerical.append(r[feature_list[i]])
            else:
                features_discrete.append(r[feature_list[i]])
        processed_data1.append(features_numerical)    
        processed_data2.append(features_discrete)
        
    df1 = pd.DataFrame(processed_data1)
    df1.columns = numerical_list
    mean_train = np.asarray(df1, dtype=np.float).mean(axis=0)
    std_train = np.asarray(df1, dtype=np.float).std(axis=0)
    std_train[std_train == 0] = 1
    data_train = np.asarray(df1, dtype=np.float64)
    norm_train = (data_train - mean_train)/(std_train)  # normalize train/validation set by substracting train mean and dividing by train std
    
    df2 = pd.DataFrame(processed_data2)
    df2.columns = discrete_list
    df_encode = pd.get_dummies(df2, columns=discrete_list, drop_first=True)
    data_encode = np.asarray(df_encode, dtype=np.float64) # one-hot encoding categorical features

    for j in range(len(data)):
        transit = data[j]
        norm_train_j=norm_train[j].tolist()
        data_encode_j=data_encode[j].tolist()
        norm_train_j.insert(0,1.0)                      # insert X0=1 into feature
        norm_train_j.extend(data_encode_j)              # concatenate numerical and categorical features
        transit['features']=norm_train_j
    return data

#Tune your parameters for final submission
#according to attached Times_Accuracy figure
#when learing rate = 0.1 and lammda = 0.001, the accuracy is highest
def submission(data):
    return train(data, 10, 0.1, 0.001)
