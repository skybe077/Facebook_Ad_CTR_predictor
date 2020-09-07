# Feature Extraction: Ad Image, Words, Target Parameters

The Facebook Ad CTR Predictor is dependent on features (as with all datasets) to use as predictors. It's not particularly difficult to break down an ad into its component parts. As shown in the image below. 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/features.JPG "Features to Extract and Make Sense of")

It's fairly simple as a concept. But finding meaningful ways to extract and then make sense of the extractions are harder questions. 

So, for each of the components, I'll outline how it was done with big ol' graphics too. 


## Making Sense of Ad Images

Intuitively and annecdotely, images make an ad. They're the first hook for an unwitting browser: stop mindless scroll to click on the ad. The question that we grappled with: What is it about an ad image that mattered to these people? 

To be honest, we didn't know. We only had vague answers like "People tend to click more on ads with faces."; "It's got to be localised to the region -- Asian faces for Asian places." Logically, we came to a conclusion -- maybe it's the items that matter in the image. Hence we used !.[Image.AI's] (http://imageai.org/) REsNet model for object detection. The choice was simply to try trained image recognition models; instead of creating our own CNN models. 

Of course, now we know better.
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/Images.JPG "Image.AI object recognition of pre-trained objects in an adset")

**Improvements to make**
1. Instead of recognising objects; try using [Saliency Maps](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/saliency%20maps.jpg)
2. Perhaps add on an [emotion detection](https://azure.microsoft.com/en-us/blog/face-and-emotion-detection/) function. 

**Code source**
* Function: detectObject_image (Functions.Features.encode_dims.py)
* Called: line 110 (01 - Clean_Transform.py)

## Making Sense of Words 
Ads aren't just made up of images. There are words too. In keeping with the principle of keeping it simple *(and getting it out quickly)*. We opted to look at individual words (1 token length); remove punctuation, formating and urls; and use TF-IDF -- where less frequent words are given more importance. Brand names, titles and such are kept in play -- annecdotal accounts suggest that audiences pay attention to brand names and titles.   

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/Words.JPG "Make sense of words in ad copy")

**Improvements to make**
1. Make a distinction between headlines and body text. Weight the  The former is commonly the first thing that people see. 
2. Remove stopwords. Not just the very light touch that we've used.   
3. Score ads based on their sentiment (see this article on [adpresso](https://adespresso.com/blog/facebook-ad-copy-sentiment-analysis/))
4. Run co-occurences instead of just tokens. 

**Code source**
* Function: detectObject_image (Functions.Features.ad_body_nlp.py)
* Called: line 67 to 80 (01 - Clean_Transform.py)


## Untangling Target Audience Parameters  

Facebook has many targeting options: education, work positions, interests, places, locations etc. Selecting the right set of targeting parameters will make or break great ad performance -- at least, that's the accepted opinion. We took adset targeting and extracted interests, work positions, and countries. As preparation, we 1-hot encoded everything in each category. 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/Parameters.JPG "Make sense of Facebook's Targeting Parameters")

**Improvements to make**
1. Distinction is too granular. CIO, CTO, Chief Technical Officers are the same. Might be useful to aggregate similar work positions, or interests.  

**Code source**
* Function: encode (clean_parameters.py). Uses a slew of helper functions.
* Called: line 52 to 56 (01 - Clean_Transform.py)

## In a Nutshell
As you can see, the extractions & meaning making aren't quite difficult -- just tedious. Regardless there's lots of room for improvement. Particularly in image detection and semantic analytics of advertising content. Out of all these feature categories, we found that text mining ads provided the greatest boost in predicting ad performance (R2 of 0.246 without text analysis vs 0.493 with text analysis)  

As we move onto the next part, [Predicting Facebook Ad CTR](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/Predicting%20CTR.md), you'll see how we used a tool (not just Python) to help us predict ad performance. 

**Dataset:** [FB Dataset.rar](https://github.com/skybe077/Facebook_Ad_CTR_predictor/tree/master/Datasets)

Specs: 2,998 rows X 5,155 columns of extracted and transformed features 

Results: Cost per unique link click	| CTR (link click-through rate)

The dataset contains
1. Ads with the following objectives (LINK_CLICKS, LEAD_GENERATION, CONVERSIONS)
2. Rows with CTR = 0 or blank were removed. These ads had either no budget set against the ad, or they performed so badly that no one is clicking
3. Identifiable data is removed *Sorry, policies are policies*

<< [Back to mainpage](https://github.com/skybe077/Facebook_Ad_CTR_predictor) || [Forward to Predicting Facebook Ad CTR](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/Predicting%20CTR.md) >>
