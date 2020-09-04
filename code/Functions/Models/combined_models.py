# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:16:00 2019

@author: skybe
"""


###########################
#
#   Combined Models 
#
#########
# Imports 


from kerasmultiinput.pyimagesearch import models
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import glob
import cv2


####################
#

def ctr_Model (trainAttrX, trainImagesX, testAttrX, testImagesX, trainY, testY, width, height, depth, opt="Adam", e1=15 , bs=128):
    print("\tCreating model...")
        
    # create the MLP and CNN models
    mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
    cnn = models.create_cnn(width, height, depth, regress=False)
    
    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn.output])
    
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="sigmoid")(x)
    
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the house)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)

    opt1 = ""
    if opt=="Adam":
        opt1 = Adam(lr=1e-3, decay=1e-3 / 200)

    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt1, metrics=['accuracy'])

    print("\tTraining model...")
    model.fit([trainAttrX, trainImagesX], trainY,validation_data=([testAttrX, testImagesX], testY), epochs=e1 , batch_size=bs)

    print("\tPredicting house prices...")
    preds = model.predict([testAttrX, testImagesX])

    return (model, preds)
        

    
############################################################################
#
#   Helper Functions
#
############################################################################

        
#############
# Load ad images & Resize them for use in mixed model
# 

def load_fb_ad_images(relevant_files, imagePath, width=1280, height= 840):
    from os import listdir
    
    files = [f for f in listdir(imagePath)]
    images = []
    
    for file in files:

        if file in relevant_files:
            basePath = imagePath + "\\" + file
            housePaths = sorted(list(glob.glob(basePath)))
            for housePath in housePaths:
                #print(housePath)
                image = cv2.imread(basePath)
                image = cv2.resize(image, (width, height))
                images.append(image)
                # print(images)
    
    # return resized image
    return np.array(images)


#############
# Normalise target variable
# 
    
def norm_target (target_array):
    
    t_info = [target_array.mean(), (target_array.max() - target_array.min())]
    target_array = (target_array - t_info[0]) / t_info[1]
    
    return (target_info, target_array)
    
#############
# Train/Test Spilt
# Basic model centers on interest + Work
    
def test_train (basic_model, images, ts = 0.7, rs=42):
    
    trainAttrX, testAttrX, trainImagesX, testImagesX = train_test_split(basic_model, images, train_size=ts, random_state=rs)
        
    return (trainAttrX, testAttrX, trainImagesX, testImagesX)

####################
#
#   assume that each array contains dataframes
    
def makeBasicDF (arrayA, arrayB, targetDF, suffixA="_A", suffixB="_B"):

    a = len(arrayA)
    b = len(arrayB)
    
    print("\tThere are "+str(a)+ " in array A | "+str(b) +"in array B. Total: " +str(a*b))
    
    basic_array = []
    
    for i in range(a):
        tempA = arrayA[i]
        tempA.add_suffix(suffixA)
        
        for x in range(b):
            tempB = arrayB[x]
            tempB.add_suffix(suffixB)
    
            temp = pd.concat([tempA, tempB, targetDF], axis=1)
            basic_array.append(temp)
            
            print("\tt Finished "+str(x)+" arrayB of "+str(i)+" arrayA. Total: "+str(a*b)+". We got "+str(a*b-i*x)+" arrays left")
   
    return basic_array
    
    
    