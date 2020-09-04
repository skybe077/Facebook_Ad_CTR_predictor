# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:30:54 2019

@author: skybe
"""

import Functions.Setup.work_dir as wd
import Functions.Features.encode_dims as ed
import Functions.Features.filter_clean as fc
import Functions.Models.correlation_models as cm

import pandas as pd

def cleanData(data, target, files):
    #############
    #   Clean data 
    #   wrk_data >> filtered data 
    #   target_param >> Target parameters (dictionary of dictionaries). We're trying to extract it all out horizontally
    #############
    wrk_data = fc.filterDataset(data, 'English', 'VIDEO')   # Get only English, non-video ads, with image urls
    print("\nClean: Filtered dataset!" )
    #wd.writeDump(wrk_data, files[1], "01 - Filtered Dataset")
    
    target_param = wrk_data[target]                         # Extract Ad Set Targeting Parameters 
    print("\nClean: Taking out the targets!" )
    #wd.writeDump(pd.DataFrame(target_param), files[1], "02 - Extracted Target Parameters")
    
    df = fc.astEval (target_param)
    print("\nClean: Evaluated targets! Prepped working list!" )
    #wd.writeDump(df, files[1], "03 - Working List")
    
    return (wrk_data, target_param, df)

def cleanData2(df, target_param, files):     
    geo_list = list(df['geo_locations'])
    flex_list = list(df['flexible_spec'])
    print("\nClean: Extracted Geo & Flex specs list!" )
    
    df_flex_spec = fc.getFlexSpecs_info(df, flex_list)      # a dataframe that contains information of flexible specs
    print("\nClean: Extracted information of Flex specs!" )
    #wd.writeDump(df_flex_spec, files[1], "04 - Flex Specs List")
    
    df = fc.getGeoLoc_info(df, target_param, geo_list)          # Geo location extraction
    print("\nClean: Extracted Geo Location!" )
    #wd.writeDump(df, files[1], "05 - Geo Location")
    
    df = fc.getAud_info (df, target_param)                      # Audience extraction
    print("\nClean: Extracted Audience Information!" )
    #wd.writeDump(df, files[1], "05 - Audience")
    
    df, df_flex_spec = fc.cleanHeaders (df, df_flex_spec)   # Clean headers of df & flex specs 
    print("\nClean: Cleaned df & df_flex_spec headers!" )
    #wd.writeDump(df, files[1], "06 - DF Headers cleaned")
    #wd.writeDump(df_flex_spec, files[1], "07 - DF Spec Headers cleaned")
    
    return (df, df_flex_spec)


def encode(df_flex_spec, files, specType, codeType, writeout):
    #############
    #   Encode data 
    #   1. 1-hot
    #   2. Feature hashing 
    #   3. PCA for 1 Variable
    #   4. PCA for 2 Variable
    #############
    
    param = df_flex_spec[specType]
    
    param_dict = ed.oneHot_param (param)
    print("\nEncode: 1-Hotted " + specType + "!")
    #wd.writeDump(pd.DataFrame(param_dict), files[1], writeout)
    
    if codeType == "featureHash":
        param_array = feat_hash(param_dict, specType)
    else:
        param_array = param_dict
            
    return (param_dict, param_array)

def feat_hash(param_dict, specType):
    p_len = int(len(param_dict[0])/2)
    param_array = ed.hashFeatures (param_dict, p_len)
    print("\nEncode: Feature Hashed " + specType + "! "+ str(p_len))
    
    return param_array

def int_corr(interest_onehotdic):
    #############
    #   Creates a Correlation Matrix and cleans duplicates
    #
    #############
    #Converts dictionary back to dataframe
    interest_onehot = pd.DataFrame.from_dict(interest_onehotdic)
    #Gets interest correlation matrix
    interest_corrmat = cm.corr_matrix(interest_onehot)
    #Cleans interest correlation matrix only
    intcorrheader = fc.headersintcorr(interest_corrmat)
    filcorr_matrix = fc.getintcorr_matrix(interest_corrmat,intcorrheader)
    #after removing duplicates, create a pure list
    purelist = fc.create_list(filcorr_matrix)
    colname = ['Type 1','Type 2','Correlation','Absolute(Correlation)']
    #create a dataframe using pure list
    intcorr_df = fc.createdf_list(purelist, len(colname), colname)
    print("\nCorrelation Matrix: Matrix cleaned for Interest ! ")
    #wd.writeDump()
    return intcorr_df

