# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:56:52 2019

@author: skybe
"""


###########################
#
#   Filtering datasets & cleaning them 
#
#

#########
# Imports 

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from imageai.Detection import ObjectDetection

##############
# Parameter Preparation
# 1. One Hot encoding fo parameters   

def oneHot_param (df):
    
    temp = pd.DataFrame()
    mlb = MultiLabelBinarizer()
    
    # for the flex_spec dataframe
    
    X = mlb.fit_transform(df)
    X = pd.DataFrame(X, columns=mlb.classes_)
    temp = pd.concat([X, temp], axis = 1)

    temp_dict = temp.to_dict('records')

    return temp_dict


##############
# Reduce Dimensionality
# 1. Feature Hashing
# 2. Principal Component Analysis (PCA)

def hashFeatures (feature_dict, features_vol = 20):

    feature_array = []
    
    for i in range(features_vol):
        h = FeatureHasher(n_features = (i+1))
        f = h.transform(feature_dict)
    
        hashed_features = f.toarray()
        hashed_features = pd.DataFrame(hashed_features)
    
        feature_array.append(hashed_features)
        print("\tWorking on feature "+str(i)+" of "+str(features_vol))

    return feature_array



##############
# Image Object Detection
# 1. Image.ai >> 
#

def detectObject_image (ad_image_input_path, ad_image_output_path, modelPath, model):
    from os import listdir

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(modelPath+"\\"+model)
    detector.loadModel(detection_speed="normal")
    
    inputfilelist = listdir(ad_image_input_path)
    
    #intialize empty dataFrame
    df = pd.DataFrame(columns=['input_image_filename', 'object_name', 'object_probability'])
    
#    start = datetime.datetime.now()
#    print("time started: ")
#    print(start.strftime("%Y-%m-%d %H:%M:%S"))
    
    index = 0
    objects_identified = 0
    
    # start going through the files
    for eachFilename in inputfilelist:
        print("\t"+eachFilename+ " detection started")

        detections = detector.detectObjectsFromImage\
        (input_image= (ad_image_input_path +"\\"+ eachFilename), \
                                  output_image_path=(ad_image_output_path +"\\"+ "resnet_normal_"+eachFilename))
        
        print("\t"+eachFilename+ " detection results:-------------------")

        for eachObject in detections:
            print("\t"+eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])

            df = df.append({"input_image_filename":eachFilename, "object_name":eachObject["name"], "object_probability":eachObject["percentage_probability"]}, ignore_index=True)
            print(eachFilename+ " detection results appended --------------------------------")
            objects_identified+=1
        index += 1
        
    print("\tDetection Completed!-------------------------------")
    print("\t"+str(index) + " images read-------------------------------")
    print("\t"+str(objects_identified) + " objects identified-------------------------------")
    
    return df
