# Human Activity Recognition Using Smartphones Sensor DataSet

## Table of Contents
- [Overview](#overview)
- [Sources/Useful Links](#sourcesuseful-links)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Which Type of ML Problem is This?](#which-type-of-ml-problem-is-this)
- [Best Performance Metrics](#what-is-the-best-performance-metric-for-this-problem)
- [Business Objectives and Constraints](#business-objectives-and-constraints)
- [Data Overview](#data-overview)
    - [1. How Data Was Recorded](#1-how-data-was-recorded)
    - [2. How is the Data Preprocessed?](#2-how-is-the-data-preprocessed)
    - [3. Y_Labels (Encoded)](#3-y_labelsencoded)
    - [4. Data Directory](#4-data-directory)
- [Train and Test Ratio](#train-and-test-ratio)
- [Agenda](#agenda)
    - [1. Analyzing the Data (EDA)](#1-analyzing-the-data-eda)
    - [2. Machine Learning Models](#2-machine-learning-models)
    - [3. Deep Learning Models](#3-deep-learning-models)
    - [4. Results & Conclusion](#4-results--conclusion)
- [Technical Aspects](#technical-aspect)
- [Installation](#installation)
- [Quick Overview of the Dataset](#quick-overview-of-the-dataset)

---

## Overview

Smartphones have become integral to daily life, equipped with advanced technology and intelligent assistance for everyday activities. They include embedded sensors such as accelerometers and gyroscopes, enabling applications based on location, movement, and context.

**Activity Recognition (AR)** refers to monitoring a person's activity using smartphone sensors. These sensors detect motion, allowing the development of systems that recognize environmental changes or specific user actions.

The **Human Activity Recognition (HAR)** framework collects raw sensor data and applies machine learning and deep learning methods to classify human movement. The UCI HAR Dataset contains data from 30 subjects performing various activities, recorded via smartphones worn at the waist. The goal is to build models that accurately predict activities such as Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying.

---

## Sources/Useful Links

- [IJRTE Blog](https://www.ijrte.org/wp-content/uploads/papers/v8i1/A1385058119.pdf)
- [Machine Learning Mastery Blog](https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/)
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Smartphone+Dataset+for+Human+Activity+Recognition+%28HAR%29+in+Ambient+Assisted+Living+%28AAL%29)

---

## Problem Statement

Given a new datapoint, predict the human activity class (e.g., Walking, Sitting, etc.) based on smartphone sensor data.

---

## Solution

The project uses both the expert-engineered and raw datasets from the UCI HAR repository:

- **Classical Machine Learning**: Uses the pre-engineered feature dataset to train models like Logistic Regression, SVM, Decision Trees, etc.
- **Deep Learning**: Uses the raw sensor data to train LSTM models for time-series classification.

---

## Which Type of ML Problem is This?

This is a **multiclass time series classification problem**. Each datapoint corresponds to one of six activities, and features are derived from sensor signal processing. Predicting human activity requires capturing temporal dependencies and distinguishing subtle differences between similar actions.

---

## What is the Best Performance Metric for This Problem?

- **Accuracy**: The primary metric used to evaluate overall model performance.
- **Confusion Matrix**: Helps understand which classes are confused and the types of errors being made.
- **Multi-class Log-loss**: Useful for evaluating multiclass classification problems.

The confusion matrix is particularly helpful in identifying confusion between similar activities (e.g., Sitting vs. Standing, Walking Upstairs vs. Walking Downstairs).

---

## Business Objectives and Constraints

- The goal is to enable health/activity tracking using only smartphone sensors (accelerometer and gyroscope).
- The solution could be implemented as a smartphone app for health monitoring.
- **Constraints**:
    - High cost of misclassification (e.g., misclassifying activity could impact health feedback).
    - No strict latency requirements.

---

## Data Overview

### 1. How Data Was Recorded

- 30 subjects performed daily activities while a smartphone (with accelerometer & gyroscope) was mounted at the waist.
- Data was recorded in windows of 2.56 seconds with 50% overlap (resulting in 1.28-second steps).
- Activities were video-recorded for accurate labeling.

**Signals recorded:**
- 3-axial linear acceleration (accelerometer): tAcc-XYZ
- 3-axial angular velocity (gyroscope): tGyro-XYZ
- Other derived signals (jerk, magnitude, frequency domain, etc.)

### 2. How is the Data Preprocessed?

- Raw sensor data was filtered and segmented into 2.56s sliding windows (128 readings per window).
- Each window yields a feature vector, computed from time & frequency domains.
- Signals are separated (e.g., body vs. gravity acceleration).
- Jerk signals and magnitudes are computed.
- FFT is applied for frequency domain features.
- **Feature functions**: mean, std, mad, max, min, sma, energy, iqr, entropy, arCoeff, correlation, maxInds, meanFreq, skewness, kurtosis, bandsEnergy, angle, etc.

**Sample features:**
- tBodyAcc-XYZ, tGravityAcc-XYZ, tBodyAccJerk-XYZ, tBodyGyro-XYZ, tBodyGyroJerk-XYZ, tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag, fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ, fBodyAccMag, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag

### 3. Y_Labels(Encoded)

| Activity            | Label |
|---------------------|-------|
| WALKING             | 1     |
| WALKING_UPSTAIRS    | 2     |
| WALKING_DOWNSTAIRS  | 3     |
| SITTING             | 4     |
| STANDING            | 5     |
| LAYING              | 6     |

### 4. Data Directory

The dataset is provided as a single ZIP file (~58 MB): **UCI HAR Dataset.zip**. The structure includes separate directories for training and test data, with expert features and raw sensor data.

---

## Train and Test Ratio

- The 30 subjects' data is split: 70% for training (21 subjects), 30% for testing (9 subjects).
- Example: `train = pd.read_csv('UCI_HAR_dataset/csv_files/train.csv')`

---

## Agenda

### 1. Analyzing the Data (EDA)

- Exploratory Data Analysis (EDA) is performed on the expert-engineered dataset.
- Data shapes: Train (7352, 564), Test (2947, 564)
- Distribution plots show that the dataset is balanced across subjects and activity classes.
- Activities are categorized as **Static** (Sitting, Standing, Laying) and **Dynamic** (Walking, Walking Upstairs, Walking Downstairs). Feature analysis (e.g., tBodyAccMagmean) helps distinguish between them.
- TSNE and boxplots are used for visualizing feature separability and activity clusters.

### 2. Machine Learning Models

**Models used:**
- **Logistic Regression:** Linear classifier; uses L2 regularization and cross-validation for hyperparameter tuning.
- **Linear SVC:** Fits a hyperplane to separate activity classes.
- **Kernel SVM:** Uses kernel functions for nonlinear separation.
- **Decision Tree:** Hierarchical, interpretable model mapping features to activity classes.
- **Random Forest:** Ensemble of decision trees, improves generalization and reduces overfitting.
- **Gradient Boosted:** Ensemble boosting method for improved accuracy (note: some systems may not support GridSearchCV with GradientBoostingClassifier due to resource limits).

### 3. Deep Learning Models

- **LSTM (Long Short-Term Memory) Networks** are applied to the raw time-series data.
    - 1-Layer LSTM
    - 2-Layer LSTM with hyperparameter tuning

### 4. Results & Conclusion

#### Classical Machine Learning Model Accuracy

| Model Name           | Features              | Hyperparameter Tuning | Accuracy Score |
|----------------------|-----------------------|-----------------------|---------------|
| Logistic Regression  | Expert features       | Yes                   | 95.83%        |
| Linear SVC           | Expert features       | Yes                   | 96.47%        |
| RBF SVM              | Expert features       | Yes                   | 96.27%        |
| Decision Tree        | Expert features       | Yes                   | 86.46%        |
| Random Forest        | Expert features       | Yes                   | 92.06%        |

**Best classical models:** Linear SVC, RBF SVM, Logistic Regression.

#### Deep Learning LSTM Model Comparison

| Model Name                   | Features           | Hyperparameter Tuning | Crossentropy | Accuracy Value |
|------------------------------|--------------------|-----------------------|--------------|---------------|
| LSTM (1 layer, 32 neurons)   | Raw time series    | Yes                   | 0.47         | 0.90          |
| LSTM (2 layers, 48/32)       | Raw time series    | Yes                   | 0.39         | 0.90          |
| LSTM (2 layers, 64/48)       | Raw time series    | Yes                   | 0.27         | 0.91          |

**LSTM models perform very well** on raw data, reaching up to 91% accuracy with two layers and appropriate tuning.

---

## Technical Aspect

This project is divided into four parts:
1. Exploratory Data Analysis (EDA)
2. Classical Machine Learning prediction models (expert features)
3. LSTM-based Deep Learning model (raw time-series data)
4. Model comparison and conclusion

---

## Installation

- The code is written in **Python 3.7**.
- Install required libraries:
    ```bash
    pip3 install pandas
    pip3 install numpy
    pip3 install scikit-learn
    pip3 install matplotlib
    pip3 install keras
    ```
- All code and scripts are available in this repository.

---

## Quick Overview of the Dataset

- **Sensors used:** Accelerometer and Gyroscope
- **Participants:** 30 volunteers
- **Activities:** Walking, Walking Upstairs, Walking Downstairs, Standing, Sitting, Laying.
- **Windowing:** 2.56s window with 50% overlap
- **Signals:** Acceleration (body & gravity), angular velocity, Jerk, and their frequency transforms
- **Features:** 561 per datapoint (means, max, mad, sma, arCoeff, energy bands, entropy, etc.)
- **Label encoding:** Each window of readings is a datapoint of 561 features with an activity label.

---

**Note:**  
- For classical ML algorithms, use expert features.
- For deep learning (LSTM), use raw sensor time-series data.

---
