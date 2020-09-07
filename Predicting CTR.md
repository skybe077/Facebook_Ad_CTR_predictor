# Predicting CTR using XGBoost & KNIME

Now that we have extracted and joined our ad features into a working dataset. It's prediction time! 

Originally, we wanted to go deep learning all the way. And developed a mixed input model that took in image data and numerical data and spat out the expected CTR. 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/mixed_model.jpg)

However the results were disappointing. R2 of 0.057944 and MSE at 0.00678. A go-through the literature suggests that our prediction models are comparatively inferior (other Mixed Input Models get R2 of 70%). This implies that we might have a fundamental issue – be it in the data preparation or transformation. 

And so we simplified our approach. 

**Trial ensemble and boosted models - XGBoost Linear Ensemble (regression) and XGBoost Tree Ensembles for regression and classification respectively.**

This was easily done up using KNIME as shown in the image below.
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/Overall.JPG)

The KNIME workflow can be downloaded from [code/knime](https://github.com/skybe077/Facebook_Ad_CTR_predictor/tree/master/code/knime). It's quite straightforward with plenty of comments. Obviously you'll need the KNIME tool to run it -- pick it up from https://www.knime.com/

## Results & Summary

Linear ensembles provided an R2 of .493 on the test dataset. It represents a far better increase in performance over Mixed Input Models.
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/results.png)

While the R2 isn’t great. However, during our test and validation phase, we got feedback from marketers that they can tolerate some inaccuracy in the predictions. Currently, this threshold is set at +/- $0.60. Upon graphing the predicted vs actual CPCs, we find that the model is suitable for use 65% of the time. In addition, it also highlights that the increase in RMSE is likely due to the lack of data for larger CPCs. 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/spread.png)

1.	This is a feature engineering problem. The bulk of time went into extraction and transformation of features from Interests, Work Titles, Countries, Objects in Ad Image, TFIDF for Ad Text. We had 5,145 features, all of which we needed. As a comparison, the current version (with text data) had better performance than earlier versions (see [Feature Extraction](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/Feature%20Extraction.md#in-a-nutshell)).
2.	As most of the features are categorical (e.g. either Interest A was set as a parameter or not), we needed to 1-hot encode them for use in the models. Interestingly enough, there was no need to reduce feature dimensionality (we did so via feature hashing and PCA, but the results were horrid). Instead using One Hot Encoding forces the model to take into account each individual feature’s impact thus generalizing the model and improving its RMSE & R2 values.
3.	Campaign objectives do matter as Facebook optimizes against these set objectives. When predicting on CPCs, we only used ads with the objective of LINK_CLICK, LEAD_GENERATION, and CONVERSION. The other objectives are simply noise.
4.	CPC distribution and skew matters. CPC is heavily skewed towards the right, with the majority of performance data falling between 0.01 to $4. Thus, the model works quite well in predicting values up to $12. But after that the model becomes unreliable.
5.	The performance of this predictor system is at R2 (0.49) & RMSE ($3.063). However, on speaking with the marketing team, they were actually OK if the system was able to predict CPCs within an offset of +/- $0.60. As such the model works about 65% of the time. 

## Last thoughts
Fascinating. 

I do not expect to compete with the ad platforms, which no doubt makes use of far greater volumes of features and ads to build far more accurate prediction models, this project has certainly shed some light on how marketers can perhaps “game” Facebook to get the most out of their marketing dollars.

The challenges faced in this project – trialing of models, many features to crunch – simply highlights that takes great effort and imagination to prepare the dataset for use. More importantly, simplicity rules over mixed models. It certainly drives home the message that features matter, and 80% of our time is spent on data wrangling and munging! 

<< [Back to mainpage](https://github.com/skybe077/Facebook_Ad_CTR_predictor)
