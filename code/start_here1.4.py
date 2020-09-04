# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 08:58:30 2019

@author: skybe
"""

########
#   1. Setup  
#   Find and setup working directories, files, and path variables
# 

import Functions.Setup.work_dir as wd

paths = wd.getWorkingDir_folders(__file__)
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
#   Dim is reduced using: 
#   1. featureHash
#############
    
# XXX_dict >> 1-hot encoded | XXX_array >> feature hashed
interests_dict, interest_array = cp.encode(df_flex_spec, files, specType="interests", codeType="featureHash", writeout="07 - Interests (1-hot)")

work_positions_dict, work_array = cp.encode(df_flex_spec, files, specType="work_positions", codeType="featureHash", writeout="08 - Work (1-hot)")

### HACK TO GET LOCATIONS ONLY
for i in range(len(df)):
    if type(df['countries'][i]) == float:
        df['countries'][i] = []

del(i)       
### END HACK 

country_dict, country_array = cp.encode(df, files, specType="countries", codeType="", writeout="08a - Countries (1-hot)")


#############
#   Detect objects in ad images 
#   Uses Resnet model
#   
############
import Functions.Features.encode_dims as ed

image_df = ed.detectObject_image (paths[3], paths[4], paths[5], nnModels[0])
wd.writeDump(image_df, files[1], "09 - Image Detections")

#pivots image df to add onto wrk_data
image_df_pv = image_df.pivot_table(index="input_image_filename",columns="object_name", values="object_probability",fill_value=0)

wrk_data1 = wrk_data.merge(image_df_pv,how="left", left_on='File Name', right_on='input_image_filename')
wrk_data1 = wrk_data1.fillna(0)

wd.writeDump(wrk_data1, files[1], "10 - Image + Work Data")

###############
#   Predict Targeting Parameters 
#   Using: 
#   1. Interest & Work vs CPC/CTR
#   2.len(work_positions_dict)
#   3. 
#############
CPC = wrk_data['Cost per unique link click']
CTR = wrk_data['CTR (link click-through rate)']

### 
#Linear regressors on target parameters: Interest, Work
# work_cpc_reg, work_cpc_values =
# interest_cpc_reg, interest_cpc_values = 
# 

w1 = []
w1.append(pd.DataFrame(work_positions_dict))
i1 = []
i1.append(pd.DataFrame(interests_dict))
im = []
im.append(wrk_data1.iloc[:,-52:])
cnt = []
cnt.append(pd.DataFrame(country_dict))

cp.predictReg(CPC, work_array, files, writeout="fh - work CPC", normalise=True)  
cp.predictReg(CPC, interest_array, files, writeout="fh - interest CPC", normalise=True) 

cp.predictReg(CPC, w1, files, writeout="all - work CPC", normalise=True)
cp.predictReg(CPC, i1 , files, writeout="all - interests CPC", normalise=True)
cp.predictReg(CTR, im , files, writeout="all - images CPC", normalise=True)
cp.predictReg(CTR, cnt , files, writeout="all - countries CPC", normalise=True)

# CTR
cp.predictReg(CTR, work_array, files, writeout="fh - work CTR", normalise=True)
cp.predictReg(CTR, interest_array, files, writeout="fh - interest CTR", normalise=True)

cp.predictReg(CTR, w1, files, writeout="all - work CTR", normalise=True)
cp.predictReg(CTR, i1 , files, writeout="all - interests CTR", normalise=True)
cp.predictReg(CTR, im , files, writeout="all - images CTR", normalise=True)
cp.predictReg(CTR, cnt , files, writeout="all - countries CTR", normalise=True)

##### 
# Interest: 48, 59
# Work: 18, 28
# im 
# cnt 
#
# Run permutations of these 4 variables 
# HACKY APPROACH
#####

interest48 = interest_array[48]
interest48 = interest48.add_suffix('_i')
interest59 = interest_array[59]
interest59 = interest59.add_suffix('_i')


work18 = work_array[18]
work18 = work18.add_suffix('_w')
work28 = work_array[28]
work28 = work28.add_suffix('_w')

cnt1 = pd.DataFrame(country_dict)
cnt1 = cnt1.add_suffix("_c")

## Interest + work 

iw1 = pd.concat([interest48, work18], axis=1)
iw2 = pd.concat([interest48, work28], axis=1)
iw3 = pd.concat([interest59, work18], axis=1)
iw4 = pd.concat([interest59, work28], axis=1)

iw1a = []
iw1a.append(pd.DataFrame(iw1))
iw2a = []
iw2a.append(pd.DataFrame(iw2))
iw3a = []
iw3a.append(pd.DataFrame(iw3))
iw4a = []
iw4a.append(pd.DataFrame(iw4))

cp.predictReg(CTR, iw1a, files, writeout="Interest48-work18 CTR", normalise=True)
cp.predictReg(CTR, iw2a, files, writeout="Interest48-work28 CTR", normalise=True)
cp.predictReg(CTR, iw3a, files, writeout="Interest59-work18 CTR", normalise=True)
cp.predictReg(CTR, iw4a, files, writeout="Interest59-work28 CTR", normalise=True)

cp.predictReg(CPC, iw1a, files, writeout="Interest48-work18 CPC", normalise=True)
cp.predictReg(CPC, iw2a, files, writeout="Interest48-work28 CPC", normalise=True)
cp.predictReg(CPC, iw3a, files, writeout="Interest59-work18 CPC", normalise=True)
cp.predictReg(CPC, iw4a, files, writeout="Interest59-work28 CPC", normalise=True)

## work + countries 
wc1 = pd.concat([work18, cnt1], axis=1)
wc2 = pd.concat([work28, cnt1], axis=1)

wc1a = []
wc1a.append(pd.DataFrame(wc1))
wc2a = []
wc2a.append(pd.DataFrame(wc1))

cp.predictReg(CTR, wc1a, files, writeout="work18-Country CTR", normalise=True)
cp.predictReg(CTR, wc2a, files, writeout="work28-Country CTR", normalise=True)

cp.predictReg(CPC, wc1a, files, writeout="work18-Country CPC", normalise=True)
cp.predictReg(CPC, wc2a, files, writeout="work28-Country CPC", normalise=True)


## interest + countries 
ic1 = pd.concat([interest48, cnt1], axis=1)
ic2 = pd.concat([interest48, cnt1], axis=1)

ic1a = []
ic1a.append(pd.DataFrame(ic1))
ic2a = []
ic2a.append(pd.DataFrame(ic1))

cp.predictReg(CTR, ic1a, files, writeout="Interest48-Country CTR", normalise=True)
cp.predictReg(CTR, ic2a, files, writeout="Interest59-Country CTR", normalise=True)

cp.predictReg(CPC, ic1a, files, writeout="Interest48-Country CPC", normalise=True)
cp.predictReg(CPC, ic2a, files, writeout="Interest59-Country CPC", normalise=True)

## interest + work + countries 
iwc1 = pd.concat([interest48, work18, cnt1], axis=1)
iwc2 = pd.concat([interest48, work28, cnt1], axis=1)
iwc3 = pd.concat([interest59, work18, cnt1], axis=1)
iwc4 = pd.concat([interest59, work28, cnt1], axis=1)


iwc1a = []
iwc1a.append(pd.DataFrame(iwc1))
iwc2a = []
iwc2a.append(pd.DataFrame(iwc2))
iwc3a = []
iwc3a.append(pd.DataFrame(iwc3))
iwc4a = []
iwc4a.append(pd.DataFrame(iwc4))

cp.predictReg(CTR, iwc1a, files, writeout="Interest48-work18-Country CTR", normalise=True)
cp.predictReg(CTR, iwc2a, files, writeout="Interest48-work28-Country CTR", normalise=True)
cp.predictReg(CTR, iwc3a, files, writeout="Interest59-work18-Country CTR", normalise=True)
cp.predictReg(CTR, iwc4a, files, writeout="Interest59-work28-Country CTR", normalise=True)

cp.predictReg(CPC, iwc1a, files, writeout="Interest48-work18-Country CPC", normalise=True)
cp.predictReg(CPC, iwc2a, files, writeout="Interest48-work28-Country CPC", normalise=True)
cp.predictReg(CPC, iwc3a, files, writeout="Interest59-work18-Country CPC", normalise=True)
cp.predictReg(CPC, iwc4a, files, writeout="Interest59-work28-Country CPC", normalise=True)

##########################################################################################
#
# Combined Model: MLP + CNN --> FC --> Regressor
# we won't know the impressions or other performance measures in advance. 
#   However we do know the targeting paramaeters. 
#   In this case: Interest, Work Pos (+ Image objects)
#   The basic model should capture parameters 
#   CNN should recognise on images 
#   As an algo:
#       1. Concat I with W (after naming them)
#       2. Concat with normalised target variables (CTR & CPC)
#       3. Test-train spilt
#       4. Build Fit Predict model(s)
#       5. Write results to 
#
#   Things to trial:
#   a. I & W Features (Reduced Dim or All) est. 25,755 dataframes
#   b. Image sizing 
#   c. optimiser & model architecture


import Functions.Models.combined_models as cm

width = 320
height = 210
depth = 3
optimiser = "Adam"
epoch = 30
batch_size = 128

images_list = cm.load_fb_ad_images(list(wrk_data['File Name']), paths[3], width, height)

#CPC1_info, CPC1 = cm.norm_target(CPC)
#CTR1_info, CTR1 = cm.norm_target(CTR)
#targetDF = pd.concat([CPC1,CTR1], axis=1)

targetDF = pd.concat([CPC,CTR], axis=1)

# Get the best performing from linear regressors
param_array = [iwc1, iwc2, iwc3, ic1, ic2]
basic_list = []

for i in range(len(param_array)):
    
    basic_df = pd.concat([param_array[i], targetDF], axis=1)
    basic_list.append(basic_df)
    
del(basic_df, i)

# m_array[basic_list][CPC or CTR][1][prediction or test result]
m_array=[]

results = paths[2]+"\\Results_CombinedModel.xlsx"    # Results Combined data dump

for i in range(len(basic_list)):
    print("\nTest-Train Spilt: item "+str(i)+" of "+str(len(basic_list))) 
    
    trainAttrX, testAttrX, trainImagesX, testImagesX = cm.test_train (basic_list[i], images_list, ts = 0.7, rs=42)

    print("\nSetting Target Variables") 
    trainCPC = trainAttrX[CPC.name] 
    testCPC = testAttrX[CPC.name] 

    trainCTR = trainAttrX[CTR.name] 
    testCTR = testAttrX[CTR.name] 

    trainTempX = trainAttrX.drop(columns=[CPC.name, CTR.name])
    testTempX = testAttrX.drop(columns=[CPC.name, CTR.name])
            
    print("\nFitting model + predicting results for CPC")    
    m, pred = cm.ctr_Model (trainTempX, trainImagesX, testTempX, testImagesX, trainCPC, testCPC, width, height, depth, opt=optimiser, e1=epoch, bs=batch_size)
    
    print("\tWriting predictions + tests for CPC")    
    wd.writeDump(pd.DataFrame(pred), results, str(i)+"-pred-CPC")
    wd.writeDump(pd.DataFrame(testCPC), results, str(i)+"-test-CPC")
    pred = [pred, testCPC]
    
    print("\nFitting model + predicting results for CTR")    
    m1, pred1 = cm.ctr_Model (trainTempX, trainImagesX, testTempX, testImagesX, trainCTR, testCTR, width, height, depth, opt=optimiser, e1=epoch, bs=batch_size)

    print("\tWriting predictions + tests for CTR")    
    wd.writeDump(pd.DataFrame(pred1), results, str(i)+"-pred-CTR")
    wd.writeDump(pd.DataFrame(testCTR), results, str(i)+"-test-CTR")
    pred1 = [pred1, testCTR]
    
    m_array.append(([m,pred],[m1,pred1]))

