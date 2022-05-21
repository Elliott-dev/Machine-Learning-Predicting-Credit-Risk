# Machine-Learning--Predicting-Credit-Risk

In this assignment, I will be building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

i will be using this data to create machine learning models to classify the risk level of given loans. Specifically, I will be comparing the Logistic Regression model and Random Forest Classifier.

## Instructions

### Retrieving the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

I will be using an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Created a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, created a testing set from the 2020 loans, also using `pd.get_dummies()`.

## Consider the models

I will be creating and comparing two models on this data: a logistic regression, and a random forests classifier. Before I create, fit, and score the models, I make a prediction as to which model I think will perform better. 

## Fitting a LogisticRegression model and RandomForestClassifier model

Created a LogisticRegression model, fit it to the data, and print the model's score. Did the same for a RandomForestClassifier. 

## Revisiting the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. Used `StandardScaler` to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, made another prediction about how Ithink scaling will affect the accuracy of the models. Wrote my predictions down and provided justification.
