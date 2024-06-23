# Leptospirosis Detection

This repository contains the code and data for detecting leptospirosis using machine learning models. The primary models used are Logistic Regression and Random Forest Classifier. The data is preprocessed and various hyperparameter tuning techniques are applied to achieve optimal performance.

## Dataset

The dataset used for this project is `train.csv`, which includes multiple features related to patient information and clinical data. Dataset `test.csv` is predicted upon the built model from the train dataset.

## Project Structure

- `scripts/`: Folder containing Jupyter Notebooks named `Leptospirosis_Detection_LR.ipynb` and `Leptospirosis_Detection_RF.ipynb` containing the complete workflow for data loading, preprocessing, model training and evaluation using Logistic Regression model and Random Forest Classification model respectively.
- `dataset/`: Folder containing training data (`train.csv`) and test data (`test.csv`) along with the dataset information in the `dataset_description.xlsx` file
- `output_csv/`: Folder containing datasets which are products from the execution of jypyter notebooks, named `Preprocessed_test_LR.csv`, `Preprocessed_test_RF.csv` containing preprocessed test data according to the format of train dataset and also `Predictions_LR.csv` and `Predictions_RF.csv` containing predictions of test data according to each of the models.
- `Leptospirosis_Detection_LR_Content.pdf` and `Leptospirosis_Detection_RF_Content.pdf`: Files containing the whole project for ease of use.

## Installation
To run this project, Python should be installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage
Open the Jupyter Notebook leptospirosis-detection-lr.ipynb or leptospirosis-detection-lr.ipynb to follow the step-by-step process of the project:

- Load the train and test datasets.
- Preprocess the data.
- Train a Logistic Regression model and a Random Forest Classifier on the train data.
- Perform hyperparameter tuning using GridSearchCV.
- Predict results on the test data.
- Evaluate the models using accuracy, confusion matrix, and classification report.

## Results
Train data is used to both train the data and also predict Leptospirosis while test data is used for prediction afterwards.

1. The best model achieved an accuracy of 89.72% (on train data) using a Logistic Regression with the following parameters:
C: 0.17
max_iter: 900
solver: newton-cg

2. The best model achieved an accuracy of 98.14% (on train data) using a Random Forest Classifier with the following parameters:
- max_depth: 23
- min_samples_leaf: 1
- min_samples_split: 4
- n_estimators: 174

Note that the test data is not checked for accuracy. 
