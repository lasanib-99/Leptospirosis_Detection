# Leptospirosis_Detection

This repository contains the code and data for detecting leptospirosis using machine learning models. The primary models used are Logistic Regression and Random Forest Classifier. The data is preprocessed and various hyperparameter tuning techniques are applied to achieve optimal performance.

## Dataset

The dataset used for this project is `train.csv`, which includes multiple features related to patient information and clinical data.

## Project Structure

- `scripts/`: Folder containing Jupyter Notebooks named `Leptospirosis_Detection_LR.ipynb` and `Leptospirosis_Detection_RF.ipynb` containing the complete workflow for data loading, preprocessing, model training and evaluation using Logistic Regression model and Random Forest Classification model respectively.
- `dataset/`: Folder containing training data (`train.csv`) and test data (`test.csv`) along with the dataset information in the `dataset_description.xlsx` file
- `output_csv/`: Folder containing datasets which are products from the execution of jypyter notebooks, named `Preprocessed_test_LR.csv`, `Preprocessed_test_RF.csv` containing preprocessed test data according to the format of train dataset and also `Predictions_LR.csv` and `Predictions_RF.csv` containing predictions of test data according to each of the models.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
