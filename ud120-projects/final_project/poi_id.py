#!/usr/bin/python

import tester
from sklearn.feature_selection import SelectKBest
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, scale
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np
import sys
import pickle
sys.path.append("../tools/")

# Select what features we will use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
finance_features = ['salary',
                    'bonus',
                    'long_term_incentive',
                    'deferred_income',
                    'deferral_payments',
                    'loan_advances',
                    'other',
                    'expenses',
                    'director_fees',
                    'total_payments']

stock_features = ['exercised_stock_options',
                  'restricted_stock',
                  'restricted_stock_deferred',
                  'total_stock_value']

email_features = ['to_messages',
                  'from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']

new_features = ['to_poi_ratio',
                'from_poi_ratio',
                'shared_poi_ratio',
                'bonus_to_salary',
                'bonus_to_total']

features_list = ['poi'] + finance_features + stock_features + email_features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Create a dataframe from the dict for cleaning and fixing data
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)
df = df[features_list]

# Fill finance missing values with zeros
df[finance_features] = df[finance_features].fillna(value=0)
df[stock_features] = df[stock_features].fillna(value=0)

# Fill email missing values with the mean 
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

df_poi = df[df['poi'] == True]
df_not_poi = df[df['poi'] == False]

df_poi.ix[:, email_features] = imputer.fit_transform(
    df_poi.ix[:, email_features])
df_not_poi.ix[:, email_features] = imputer.fit_transform(
    df_not_poi.ix[:, email_features])
df = df_poi.append(df_not_poi)

# Fix the errors in the data 
# Shift left Belfer data by one column
belfer_finance = df.ix['BELFER ROBERT', 1:15].tolist()
belfer_finance.pop(0)
belfer_finance.append(0)
df.ix['BELFER ROBERT', 1:15] = belfer_finance

# Shift right Bhatnagar data by one column
bhatnagar_finance = df.ix['BHATNAGAR SANJAY', 1:15].tolist()
bhatnagar_finance.pop(-1)
bhatnagar_finance = [0] + bhatnagar_finance
df.ix['BHATNAGAR SANJAY', 1:15] = bhatnagar_finance

# Drop the outliers
df.drop(axis=0, labels=[
        'TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
df.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J',
        'WHALLEY LAWRENCE G', 'BAXTER JOHN C'], inplace=True)

# Add new features to dataframe
df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments']
df.fillna(value=0, inplace=True)

# Scale the data frame
scaled_df = df.copy()
scaled_df.ix[:, 1:] = scale(scaled_df.ix[:, 1:])

# Create my_dataset
my_dataset = scaled_df.to_dict(orient='index')


clf = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier(criterion='entropy',
                                        max_depth=None, max_features=None, min_samples_split=20))
])

features_list = ['poi'] + email_features + finance_features + stock_features + new_features

dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()
