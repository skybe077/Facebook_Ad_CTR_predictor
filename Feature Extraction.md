# Feature Extraction: Ad Image, Words, Target Parameters

The Facebook Ad CTR Predictor is dependent on features (as with all datasets) to use as predictors. It's not particularly difficult to break down an ad into its component parts. As shown in the image below. 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/features.JPG "Features to Extract and Make Sense of")

It's fairly simple as a concept. But finding meaningful ways to extract and then make sense of the extractions are harder questions. 

So, for each of the components, I'll outline how it was done with big ol' graphics too. 


## Making Sense of Ad Images

Intuitively and annecdotely, images make an ad. They're the first hook for an unwitting browser: stop mindless scroll to click on the ad. The question that we grappled with: What is it about an ad image that mattered to these people? 

To be honest, we didn't know. We only had vague answers like "People tend to click more on ads with faces."; "It's got to be localised to the region -- Asian faces for Asian places." Logically, we came to a conclusion -- maybe it's the items that matter in the image. Hence we used !.[Image.AI's] (http://imageai.org/) YOLOv3 model for object detection. The choice was simply to try trained image recognition models; instead of creating our own CNN models. 

Of course, now we know better.
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/Images.JPG "Image.AI Image Extraction")

**Improvements to make**
1. Instead of recognising objects; try using Saliency Maps 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/saliency%20maps.jpg)
src: [Deep CTR Prediction in Display Advertising – DNN to work with sparse images](https://www.researchgate.net/publication/308364214_Deep_CTR_Prediction_in_Display_Advertising)

2. Perhaps add on an [emotion detection](https://azure.microsoft.com/en-us/blog/face-and-emotion-detection/) function. 

