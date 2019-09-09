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


<img src="Credit%20Card%20FraudDetection%20%20Class%20Count%20Plot.PNG" width="800" >






***2.4	Exploratory Data Analysis (EDA)***

Summary Statistics and Histograms of the Attributes are used to highlight the overall view of the credit card fraud detection dataset. Table 1 depicts the Summary Statistics of the dataset features. 

  
<img src="/Plots/Summary%20Statistics%20Table%201.PNG" width="800" >


The data is from two days credit card transactions as highlighted by maximum time in seconds of 172792 in Table 1 above. This is approximately equal to two days in seconds. Furthermore, the minimum transaction amount is 0.0 and the maximum transaction amount 88.34.


 Tables 2 and 3 further provide the summary statistics of fraudulent and non-fraudulent transactions. 
 
 
 <img src="/Plots/Fraudulent%20Transactions%20Summary%20Statistics%20Table%202.PNG" width="800" >

 
 <img src="/Plots/Non-Fraudulent%20Transactions%20Summary%20Statistics%20Table%203.PNG" width="800" >
 
 
    

    
    
From Table 2 above for fraudulent transaction class summary statistics, the minimum transaction amount is 0.0, the maximum 
transaction amount is 2125.87 and the average transaction amount is 122.21. 
Hence, it can be inferred that an average of 122 is lost to fraudulent transaction. However, from Table 3 above for non-fraudulent transaction class summary statistics, the minimum transaction amount is 0.0, the maximum transaction amount is 25691 and the mean value of non-fraudulent is 88.89. Hence, the average amount for non-fraudulent transaction is 88.29.  Also, based on the difference between the minimum and maximum transaction amount, it can be concluded that non-fraudulent transaction class has a wide range. Furthermore, figures 4 and 5 showcase the top 10 distribution of fraudulent and non-fraudulent transactions.


<img src="/Plots/Distribution%20of%20Top%2010%20Fraudulent%20Transaction%20Amount.PNG" width="800" >


<img src="/Plots/Distribution%20of%20Top%2010%20Non-Fraudulent%20Transaction%20Amount.PNG" width="800" >



Histograms of the dataset variables were created to visualize the distribution of the features data. Figure 6 showcases the histogram for each of the credit card fraud detection attribute. Each histogram below highlights if the distribution of data for each of the variable is symmetric, left-skewed or right-skewed. 


<img src="/Plots/Histograms%20of%20Credit%20Card%20Fraud%20Detection%20Features.PNG" width="800" >


***3.	 Feature Selection***

The performance of a classification machine learning model is greatly affected by feature selection. It is worth noting that model performance can be negatively affected by irrelevant or partially pertinent features. The process of manually or automatically selecting those features or input that contribute the most to the target or prediction variable is called Feature Selection. It minimizes overfitting, fosters prediction accuracy and minimizes training time.


To determine the important features for the transaction class, univariate selection process was leveraged in this project to identify relevant features (independent variables) that contribute the most to the classification of transactions as either non-fraudulent or fraudulent.


<img src="/Plots/Univariate%20Feature%20Selection%20Result.PNG" width="600" >


The above figure 7 result identifies features V17, V14, V12, V10 and V16 are the top five features that facilitate the classification of a transaction as either a non-fraudulent or fraudulent transaction. Feature Selection is further performed for most of the classification models developed in this project.

It is worth noting that some classifiers do have feature importance attribute. The following features importance results were produced by Decision Tree, Random Forest and Extreme Gradient Boosting Classifiers.




***4.	Methods***

***4.1 Models***

There are two types of supervised machine learning algorithms including Regression and Classification.  To classify credit card transactions as either fraudulent or non-fraudulent, predictive classification models are used. Using Python Scikit learn library, various kinds of classification models can be developed and implemented. 

For this practicum project, I chose to utilize five different Python Scikit learn classification methods/algorithms for training and building five different classification models:


•	Logistic Regression: This is a classification algorithm.  In this project, this algorithm is leveraged to estimate the probability of a credit card transaction class outcome as either fraudulent or non-fraudulent based on the 28 PCA transformed features and transaction amount. 

•	Decision Tree Classifier: This is the second classification model constructed in this project.  In this project, it trains a classification model in a flowchart-like tree structure to output predicted transaction class as either fraudulent or non-fraudulent.

•	Random Forest Classifier. This is the third model developed in this classification predictive project. It is an ensemble method and it combines multiple decision trees in determining the final output.

•	Extreme Gradient Boosting:  This is the fourth python Scikit learn library classification algorithm utilized for model construction in this project. This classification algorithm develops models in a stage-wise manner and creates a prediction model in the form of an ensemble of weak prediction models, normally decision tree.

•	 K Nearest Neighbors (KNN): This is the fifth classification algorithm leveraged for model construction in this project. KNN works by leveraging a number of nearest neighbors (k) to classify outcomes in a dataset.


***4.2	Methods***

In this project the following methods were utilized in the construction of the models:

•	All the five models were first constructed using the dataset with a single split into training and test dataset with a 70 to 30 ratio. The models were fitted on the training dataset and the performance of the models were evaluated utilizing the test data.

•	The models were developed the second time with the incorporation of 5-fold cross-validation on the dataset. This requires training the same classification model multiple (five) times leveraging different split each time.

•	The models were constructed the third time utilizing both cross-validation and undersampling techniques. In this project, undersampling was performed by randomly selecting 492 observations from the non-fraudulent class to match the number of observations in the fraudulent transaction class.

<img src="/Plots/Undersampled%20Transaction%20Class%20Distribution.PNG" width="800" >

•	The models were constructed the fourth time utilizing both cross-validation and oversampling techniques. In this project, oversampling was performed by randomly replicating observations in fraudulent transaction class to match the number of observations in the non-fraudulent class.

<img src="/Plots/Oversampled%20Transaction%20Class%20Distribution.PNG" width="800" >


***5	Results***

***5.1 Presentation of the results***

The following is the Precision, Recall, f1-score, and Precision-Recall AUC model performance metrics generated by the five models based on a single 70:30 train to test split ratio:

***Classification Models' Results - Train-Test Single Split***

<img src="/Plots/Train-Test%20Single%20Split%20Performance%20Models%20Results.PNG" width="800" >


From these Train-Test Single performance results, Extreme Gradient Boosting model based on a single 70/30 train to test ratio split produced precision score of 0.95, Recall score of 0.82, F1-Score of 0.88 and Precision-Recall AUC score of 0.89 for the Fraudulent Transaction. Hence, Extreme Gradient Boosting model produced the best metrics among the five models developed for this single train-test split. 


The followings are the Precision-Recall Curves depicting the five models Precision-Recall Curves:


