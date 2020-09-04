# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 08:58:30 2019

@author: skybe
"""
##################################################################
####Under ~/04 - All together/Functions/Features/encode_dims.py
##change "\\" to "/" for Mac users

########
#   1. Setup  
#   Find and setup working directories, files, and path variables
# 

import Functions.Setup.work_dir_val as wd

paths = wd.getWorkingDir_folders("__file__")
files = wd.getFileVariables()
meta, target, metrics = wd.getDatasetVariables ()
nnModels = wd.getModelVariables ()

files[0] = paths[1]+"\\"+files[0]   # facebook ad data
files[1]= paths[2]+"\\"+files[1]    # Step through
files[2]= paths[2]+"\\"+files[2]    # Results  data dump

#############
#   Get data 
#   
#############
import pandas as pd

data = pd.read_excel(files[0], sheet_name='FB Dataset')
#############
#   Extract Targeting Parameters 
#
#
#############
import clean_parameters as cp

wrk_data, target_param, df = cp.cleanData(data, target, files)
df, df_flex_spec = cp.cleanData2(df, target_param, files) 


#############
#   Clean & Encode work & interest
#   Creates a Correlation Matrix Within Interest
#   Only 1-hots everything
#############
    
# XXX_dict >> 1-hot encoded | XXX_array >> feature hashed
interests_dict, interest_array = cp.encode(df_flex_spec, files, specType="interests", codeType="1-hot", writeout="")

work_positions_dict, work_array = cp.encode(df_flex_spec, files, specType="work_positions", codeType="1-hot", writeout="")

country_dict, country_array = cp.encode(df, files, specType="countries", codeType="", writeout="")

### HACK TO GET LOCATIONS ONLY
for i in range(len(df)):
    if type(df['countries'][i]) == float:
        df['countries'][i] = []

del(i)       
### END HACK 

# Run TF-IDF on Ad Copy
import Functions.Features.ad_body_nlp as abn

adwords = abn.getAdwords(wrk_data, specType='Ad body')
adwords_1_hot = abn.adwords_1_hot(adwords)
adwords_col = abn.getcolname(adwords_1_hot)

adwords_countwords = abn.countwords(adwords,adwords_col)
wordcountdf = pd.DataFrame(adwords_countwords, columns = adwords_col) 
countwordindoc = abn.wordindoc(wordcountdf,adwords_col)
TFIDF = abn.computeTFIDF(adwords_countwords,countwordindoc,adwords_col)

del(adwords_countwords, wordcountdf, countwordindoc, adwords, adwords_1_hot, adwords_col)

TFIDF.to_excel("Datasets\\Adbody_Validation.xlsx", sheet_name = "sheet 1")

#############
#   Detect objects in ad images 
#   Uses Resnet model
#   
############

'''
Mac Users might get error due to '.DS_Store'
    To remove the error, go to terminal
    Change directory to the folder followed by deleting the default file.

    "cd '<YOUR PATH>/Datasets/Fb_Ad_Images'
    find . -name '.DS_Store' -type f -delete"
'''
##############################################
# Extracts images from the ad creative URLs
# Not used as Facebook tends to remove creatives from their server after some time. 
# Current solution: Download & store in database

#import Functions.Setup.get_ad_images as gai
#gai.getImages(wrk_data["Ad creative image URL"], wrk_data["Ad creative thumbnail URL"], paths[3]+"\\Ad_Image_{0}.jpg")
##############################################

import Functions.Features.encode_dims as ed
# Detects objects from the ad image database
# It is non-discriminatory & uses 
# 

image_df = ed.detectObject_image (paths[3], paths[4], paths[5], nnModels[0])

# Pivots image df to add onto wrk_data
image_df_pv = image_df.pivot_table(index="input_image_filename",columns="object_name", values="object_probability",fill_value=0)

# Only extracts images related to work data 
img_filenames = pd.DataFrame(wrk_data['File Name'])

wrk_data1 = img_filenames.merge(image_df_pv,how="left", left_on='File Name', right_on='input_image_filename')
wrk_data1 = wrk_data1.fillna(0)

image_df = wrk_data1.drop("File Name", axis=1)

del(wrk_data1, img_filenames, image_df_pv)

###############
#   Join them all together 
#   CPC, CTR, interests, work_positions, countries, images, adbody
#############
CPC = wrk_data['Cost per unique link click']
CTR = wrk_data['CTR (link click-through rate)']
obj = wrk_data['Campaign objective']

knime_df = pd.concat([CPC, CTR], axis=1, sort=False)
knime_df = pd.concat([knime_df, pd.DataFrame(interests_dict)], axis=1, sort=False)
knime_df = pd.concat([knime_df, pd.DataFrame(work_positions_dict)], axis=1, sort=False)
knime_df = pd.concat([knime_df, pd.DataFrame(country_dict)], axis=1, sort=False)
knime_df = pd.concat([knime_df, image_df], axis=1, sort=False)
knime_df = pd.concat([knime_df, pd.DataFrame(TFIDF)], axis=1, sort=False)
knime_df = pd.concat([knime_df, obj], axis=1, sort=False)

wrk_data.to_excel("Datasets/for Knime/wrk Data1.xlsx", sheet_name = "sheet 1")

# Write out to excel; partition in Knime
knime_df.to_excel("Datasets/for Knime/All Data.xlsx", sheet_name = "sheet 1")

'''
# 80 / 20 spilt
'''

Training = knime_df.sample(n=int(len(knime_df)*0.8), random_state=42)
Testing = knime_df.drop(Training.index)

Training = Training.dropna(axis=0)        
Training.to_excel("Datasets/for Knime/Training Data.xlsx", sheet_name = "sheet 1")

Testing = Testing.dropna(axis=0)        
Testing.to_excel("Datasets/for Knime/Validation Data.xlsx", sheet_name = "sheet 1")

'''
# Partition training and test set using 
# 1) 2017 to 9/2019 Data (3242)
# 2) 9 to 10/2019 Data (149)
'''

Training1 = knime_df.iloc[list(range(3242))]
Testing1 = knime_df.iloc[list(range(3242,len(knime_df)))]

Training1 = Training1.dropna(axis=0)        
Training1.to_excel("Datasets/for Knime/Training Data1.xlsx", sheet_name = "sheet 1")

Testing1 = Testing1.dropna(axis=0)        
Testing1.to_excel("Datasets/for Knime/Validation Data1.xlsx", sheet_name = "sheet 1")

