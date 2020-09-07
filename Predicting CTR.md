# Predicting CTR using XGBoost & KNIME

Now that we have extracted and joined our ad features into a working dataset. It's prediction time! 

Originally, we wanted to go deep learning all the way. And developed a mixed input model that took in image data and numerical data and spat out the expected CTR. 

![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/mixed_model.jpg, "")

However the results were disappointing. R2 of 0.057944 and MSE at 0.00678. A go-through the literature suggests that our prediction models are comparatively inferior (other Mixed Input Models get R2 of 70%). This implies that we might have a fundamental issue – be it in the data preparation or transformation. 

And so we simplified our approach. 

**Trial ensemble and boosted models - XGBoost Linear Ensemble (regression) and XGBoost Tree Ensembles for regression and classification respectively.**

This was easily done up using KNIME as shown in the image below.
![alt text](https://github.com/skybe077/Facebook_Ad_CTR_predictor/blob/master/images/Overall.JPG, "")
