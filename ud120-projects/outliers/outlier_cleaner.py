#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).
        
        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np

    cleaned_data = []
    ### your code goes here
    err_list = []  
    for i in range(len(ages)):
        err_list.append(net_worths[i] - predictions[i])
    
    for k in range(9):
        i = err_list.index(max(err_list, key=abs))
        err_list.pop(i)
        ages = np.delete(ages, i)
        net_worths = np.delete(net_worths, i)
        

    for i in range(len(ages)):
        cleaned_data.append(tuple((ages[i], net_worths[i], net_worths[i] - predictions[i])))
    
    return cleaned_data

