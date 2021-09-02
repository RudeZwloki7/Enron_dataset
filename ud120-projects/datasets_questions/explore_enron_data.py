#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(
    open("../final_project/final_project_dataset.pkl", "rb"))
print 'Count of people in dataset:', len(enron_data.keys())

count = 0
num = 0
for k in enron_data.keys():
    if enron_data[k]['poi'] == 1:
        num +=1
    if enron_data[k]['total_payments'] == 'NaN' and enron_data[k]['poi'] == 1:
        # print enron_data[k]
        count+=1
print 'All poi', num
print 'Count of poi NaN total_payments', count

# print enron_data['SKILLING JEFFREY K']['total_payments']
# print enron_data['LAY KENNETH L']['total_payments']
# print enron_data['FASTOW ANDREW S']['total_payments']

