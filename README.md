# PracticumII--Classification-Predictive-Models-for-Credit-Card-Fraud-Detection
MSDS-696-Data Science Practicum II Project - To build 5 machine learning classification predictive models and select the best model to leverage for credit card fraud detection.




***1.	Introduction***

Credit Card Fraud Detection is a challenging task. This is due to several features that should be examined for accurate and timely detection of Credit Card Fraudulent Transactions. It is imperative that credit card companies including banking, finance, retail, and e-commerce companies can identify fraudulent credit card transactions so that customers are not charged for unauthorized transactions. According to legal dictionary, "Credit Card Fraud is the unauthorized use of an individual's credit card information to make purchases, or to remove funds from the cardholder's account".
The purpose of this data science practicum project is to develop a machine learning classification predictive model that can best be leveraged in credit card fraud detection. The second motivation behind this project is to create a classification machine learning model that can be utilized by credit card companies to significantly prevent the loss of billions of dollars to credit card fraudulent transactions.




***2.	Data***

***2.1 Presentation of the data***

This project used Kaggle data set located at https://www.kaggle.com/mlg-ulb/creditcardfraud to achieve its objectives. It contains transactions that took place in two days, where out of 284,807 transactions we have 492 fraudulent transactions. The dataset contains only numerical input variables due to Principal Component Analysis (PCA) transformation and confidentiality issues. The dataset has 31 Features including 28 PCA transformed Attributes and two attributes that are not transformed by PCA. The last feature is the target attribute of interest in this practicum project. This last feature is the Transaction Class and it can be Fraudulent or Non-Fraudulent.

***2.2 Data Preparation***

There are no missing values identified as NaNs/Null and zeros (0.0) in the dataset used for this project. Missing values if not dealt with would result in bias resulting from the differences between missing and complete data. 

***2.3 The Imbalanced Transaction Class Distribution***

It is imperative to highlight the significant contrast within the Transaction Class. As expected, most of the transactions are Non-Fraudulent while only very few transactions are Fraudulent Transactions. 
