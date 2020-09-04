# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:02:04 2019

@author: skybe
"""

###########################
#
#   Filtering datasets & cleaning them 
#
#

#########
# Imports 


import ast #abstract syntax trees package | used to convert string of 'Ad set targeting' to dictionary
import pandas as pd
import numpy as np # used in geo location


# filters dataset by language, ad type. Only returns ads with URL
def filterDataset(ds, language, adType):
    
    # Filter out records that are not English
    is_Lang =  ds['Language']==language
    ds = ds[is_Lang]
    
    # Filter out records that are videos
    not_type = ds['Ad creative object type']!=adType
    ds= ds[not_type]
    
    # Get only data that have the ad creative image URL
    ds = ds[ds['Ad creative image URL'].notnull()]
    ds = ds.reset_index()
    ds = ds.drop(['index'], axis=1)
    image_names = ds['File Name']

    return ds

#### convert string of 'Ad set targeting' to dictionary
# SLOW!!! Because it reads ad set targets
# This part is needed to create the working df for 1-hot & dimensionality reduction
 
def astEval (target_param):
    
    print ("Start Ast")
    
    for i in range(len(target_param)):
        test = ast.literal_eval(target_param[i])
        target_param[i] = test

    print ("Finished Ast")
    
    # set working_list into a dataset
    working_list = []
    
    for i in target_param:
        working_list.append(i)

    return pd.DataFrame(working_list)



############
# Facebook specific functions. 
# 1. getFlexSpecs_info >> Extracts info from flexible specs
# 2. getGeoLoc_info >> 
# 3. getAud_info
    
def getFlexSpecs_info (df, flexible_spec_list):
    
    for i in range(len(df)):   
        if type(flexible_spec_list[i]) == float:        # checks if it's empty; if so then give it an empty INTEREST list
            flexible_spec_list[i] = {'interests': 0}
            continue
        flexible_spec_list[i] = flexible_spec_list[i][0] # Extracts only interests[0]
                                                         # Other things are education and work positions 
        
    df_flex_spec = pd.DataFrame(flexible_spec_list)
    
    ####df_flex_spec is used to extract all data from flexible specs (10):  
    #       ['behaviors', 'education_majors', 'education_schools',
    #       'education_statuses', 'family_statuses', 'industries', 'interests',
    #       'life_events', 'work_employers', 'work_positions']
    
    ## get the headers
    headers = df_flex_spec.columns.values
    # instead of a dict of id and names, replace it as a list of names
    for i in flexible_spec_list:
        for header in headers:
            #check if key is in here
            if header not in i:
                continue
            #if input value is 0 (means nothing in there), continue
            if type(i[header]) == int:
                continue
            temp_list = []
            for j in i[header]:
                # value of education status is already in list form (not dict form)
                if type(j) ==int:
                    break
                temp_list.append(j['name'])
            #print(temp_list)
            i[header] = temp_list
    
    df_flex_spec = pd.DataFrame(flexible_spec_list)
    df_flex_spec['interests'].replace(0, np.nan, inplace=True)
    
    return df_flex_spec

######################################################################
# structuring the geo_locations column --> add new columns countries and location_types
# need to remove geo_locations column but havent yet (just in case)
def getGeoLoc_info (df, wrk_data, geo_locations):
    
    geo_locations = {'geo_locations': []}
    # creates a new column for both countries and location_types
    df['location_types'] = pd.Series(np.random.randn(100))
    df['countries'] = pd.Series(np.random.randn(100))
    
    for i in range(len(wrk_data)):
        if ('geo_locations' in wrk_data[i].keys()):
            geo_locations['geo_locations'].append(wrk_data[i]['geo_locations'])
        else:
            geo_locations['geo_locations'].append(0)
    
    for i in range(len(df)):  
        if type(df['geo_locations'][i]) == float:
            continue
        if 'countries' in df['geo_locations'][i]:
            df['countries'][i] = geo_locations['geo_locations'][i]['countries'] # extracts 1 country
        elif 'cities' in df['geo_locations'][i]:
            df['countries'][i] = list(geo_locations['geo_locations'][i]['cities'][0]['country'])    # extracts list of cities
        if 'location_types' in df['geo_locations'][i]:
            df['location_types'][i] = geo_locations['geo_locations'][i]['location_types']   # extracts locations
            
    df = cleanCountry(df)           
    
    return df

def cleanCountry(df):
    for i in range(len(df)):
        if type(df['countries'][i]) == float:
            df['countries'][i] = []
        if type(df['location_types'][i]) == float:
            df['location_types'][i] = []
    return df

# structuring the custom_audiences column and the excluded_custom_audiences column also

def getAud_info (df, wrk_data):
    c_audience = {'c_audience': []}
    ec_audience = {'ec_audience': []}
    
    for i in range(len(df)):
        if 'custom_audiences' in wrk_data[i]:
            c_audience['c_audience'].append(wrk_data[i]['custom_audiences'])
        else:
            c_audience['c_audience'].append(0)
        if 'excluded_custom_audiences' in wrk_data[i]:
            ec_audience['ec_audience'].append(wrk_data[i]['excluded_custom_audiences'])
        else:
            ec_audience['ec_audience'].append(0)
    
    # instead of a dict of id and interest/work positions, replace it as a list of interests/ work positions
    for i in range(len(c_audience['c_audience'])):
        if c_audience['c_audience'][i] != 0:
            c_audience_list = []
            for j in range(len(c_audience['c_audience'][i])):
                c_audience_list.append(c_audience['c_audience'][i][j]['name'])
            c_audience['c_audience'][i] = c_audience_list
        if ec_audience['ec_audience'][i] != 0:
            ec_audience_list = []
            for j in range(len(ec_audience['ec_audience'][i])):
                ec_audience_list.append(ec_audience['ec_audience'][i][j]['name'])
            ec_audience['ec_audience'][i] = ec_audience_list
            
    for i in range(len(df)):
        df['custom_audiences'][i] = c_audience['c_audience'][i]
        df['excluded_custom_audiences'][i] = ec_audience['ec_audience'][i]    
        
    return df        

# Changing headers 
def cleanHeaders (df, df_flex_spec):

    for header in df_flex_spec.columns.values:
        df_flex_spec.loc[df_flex_spec[header].isnull(),[header]] = df_flex_spec.loc[df_flex_spec[header].isnull(),header].apply(lambda x: [])
    
    # replace in df
    for header in df.columns.values:
        df.loc[df[header].isnull(),[header]] = df.loc[df[header].isnull(),header].apply(lambda x: [])

    return (df, df_flex_spec)
