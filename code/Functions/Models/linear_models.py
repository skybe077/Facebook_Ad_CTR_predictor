# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:20:30 2019

@author: skybe
"""

###########################
#
#   Linear Models 
#
#########
# Imports 

from sklearn import linear_model as lm, model_selection as ms, metrics as m 


###############################################
# Regression training & testing 
# http://www.fairlynerdy.com/what-is-r-squared/

def adLinear (train_x, test_x, train_y, test_y ):
    
    reg = lm.LinearRegression()
    # Train the model using the training sets
    reg.fit(train_x, train_y)
    
    # Make predictions using the testing set
    pred_y1 = reg.predict(train_x)
    pred_y2 = reg.predict(test_x)
    
    # The coefficients
    # print('Coefficients: \n', reg.coef_)
    # The mean squared error
#    print("Mean squared error: %.2f" % m.mean_squared_error(test_y, pred_y2))
    # Explained variance score: 1 is perfect prediction
#    print('Variance score: %.2f' % m.r2_score(test_y, pred_y2))
    
    return (reg, [m.mean_squared_error(train_y, pred_y1), m.r2_score(train_y, pred_y1), m.mean_squared_error(test_y, pred_y2), m.r2_score(test_y, pred_y2)])


# helper function for testing across multiple features
# needs input as an array of Dataframes
def getLinearReg (dataArray, targetArray, ts=0.7, rs=42):    
    regArray = []
    valuesArray = []
    total = len(dataArray)
    
    # Predict on X positions
    for x in range(total):
    
        rArray = ms.train_test_split(dataArray[x], targetArray, train_size=ts, random_state=rs)    #CPC_array = dfCPC_train_x, dfCPC_test_x, dfCPC_train_y, dfCPC_test_y
    
        values = adLinear (rArray[0], rArray[1], rArray[2], rArray[3])
        regArray.append(values[0])
        valuesArray.append(values[1])   # Try this one
        print ("\tLinear Regressed on "+str(x)+" out of "+str(total))

    return [regArray, valuesArray]
