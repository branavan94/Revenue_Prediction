# Revenue_Prediction
## 🐣 Introduction
In this competition, we a’re challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer.

About the dataset:

Similar to most other kaggle competitions, we are given two datasets
Data can be found with this link : <a href="https://www.kaggle.com/c/ga-customer-revenue-prediction/data" target="_blank">Click Here</a></br>
* train.csv
* test.csv </br>
Each row in the dataset is one visit to the store. We are predicting the natural log of the sum of all transactions per user.

The data fields in the given files are</br>

* fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
* channelGrouping - The channel via which the user came to the Store.
* date - The date on which the user visited the Store.
* device - The specifications for the device used to access the Store.
* geoNetwork - This section contains information about the geography of the user.
* sessionId - A unique identifier for this visit to the store.
* socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
* totals - This section contains aggregate values across the session.
* trafficSource - This section contains information about the Traffic Source from which the session originated.
* visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, we should use a combination of fullVisitorId and visitId.
* visitNumber - The session number for this user. If this is the first session, then this is set to 1.
* visitStartTime - The timestamp (expressed as POSIX time).</br>

Also it is important to note that some of the fields are in json format.

## 🎯 Objectives

We are predicting the natural log of the sum of all transactions per user. 
Once the data is updated, as noted above, this will be for all users in test_v2.csv for December 1st, 2018 to January 31st, 2019. For every user in the test set, the target is:
Note that the dataset does NOT contain data for December 1st 2018 to January 31st 2019. 
We must identify the unique fullVisitorIds in the provided test_v2.csv and make predictions for them for those unseen months.

<a href="https://htmlpreview.github.io/?https://github.com/branavan94/Revenue_Prediction/blob/main/Project.html" target="_blank">View Code here</a>
