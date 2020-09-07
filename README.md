# Facebook Ad CTR Predictor 
Predict the Click-Through-Rate of a Facebook ad based on audience targeting parameters, Ad creatives (Image + Copy). 
Currently R2 is 0.493. Depending on threshold (+/- $0.60), this predictor was useful for 60 to 70% of the time.   


## Motivations  
Social media advertising tends to be hit and miss. Commomly media marketers wouldn't know if an ad works until its been running for 2 weeks. But that means it's 2 weeks of money gone to waste -- if the ad is low performing. What can we do to get the most bang out of limited bucks? 

All we have before a campaign runs is an ad and a set of target parameters (e.g. work positions, countries, interests). Based on that information, can we predict an Ad's performance? 

Apparently, yes. 

## Predecessors 
A Google search turns up quite a number of papers and approaches with respect to predicting advertising performance. Scroll down to the end for the links to the papers.
TLDR: 
1. Model ensembles are used to extract features from ad creatives & basic user data
2. The features are pushed into Logistic Regression to predict if an ad is clicked on or otherwise by a user
3. These approaches rely on getting access to user data and ad logs (most companies won't have access to it) 

As much as I'd like to copy and use the models, it's not going to work. I simply will never have access to user-level data that these researchers have. But there are things that we can reuse -- particularly in feature extraction. 

## How It Works 
If we don't have user-level datasets, then what can we do? What if we did it on an ad level?
Break down the ad into its components, feature engineer the heck out of them, and then run it through a linear predictor? 
Given the constraints of our available dataset, that's all we could do.   

The following image is the final solution outline based on that principle.
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/soln_outline.png "Facebook Ad CTR Predictor Solution")

It's quite straightforward. Extract from Facebook using Supermetrics; Clean and transform components of the dataset; Merge them all together; Then run it through the predictor. 

This is the overall workflow. A bit more detail on what went where.  
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/workflow.png "Facebook Ad CTR Predictor Workflow")

For easy reference, I've created the following markdowns to go through each part:
1. Feature Extraction: Ad Image, Words, Target Parameters 
2. Predicting CTR
3. Did it work? 

## Specs  
Dataset contains 510 Interests, 203 Work Positions, and 52 Image Objects (link)
Tools: Supermetrics, Python, KNIME

## Useful Links 
!.[1.	Deep CTR Prediction in Display Advertising – DNN to work with sparse images] (https://www.researchgate.net/publication/308364214_Deep_CTR_Prediction_in_Display_Advertising)

2.	Practical Lessons from Predicting Clicks on Ads at Facebook
https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf

