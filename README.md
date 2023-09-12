## Introduction
This repository contains code for building and training an Artificial Neural Network (ANN) using TensorFlow for binary classification tasks. The following sections provide a detailed overview of the code and its components.

## Part 1 - Data Preprocessing
In this section, various data preprocessing steps are performed to prepare the data for training the neural network.

### Importing Libraries
Necessary libraries such as NumPy, Pandas, and TensorFlow are imported to support the code's functionality.

### Data Loading and Exploration (EDA)
The dataset is loaded from the 'Churn_Modelling.csv' file, and Exploratory Data Analysis (EDA) is conducted. This includes checking data types, non-null counts, and examining the first few rows of the dataset. Additionally, irrelevant columns are dropped to improve data quality.

### Data Visualization
Data visualization is performed using Seaborn to gain insights into the relationships between different features and the target variable ('Exited').

### Encoding Categorical Data
To make the data suitable for machine learning, label encoding is applied to the 'Gender' column, and one-hot encoding is applied to the 'Geography' column to convert categorical data into a numerical format.

### Splitting the Dataset
The dataset is split into training and test sets to facilitate model training and evaluation.

### Feature Scaling
Standardization is applied to scale the numerical features, ensuring that they have comparable scales.

## Part 2 - Building the ANN
In this section, the Artificial Neural Network (ANN) architecture is constructed.

### Initializing the ANN
An empty sequential model is created using TensorFlow's Keras API, serving as the foundation for building the neural network.

### Adding Layers
Three layers are added to the neural network: two hidden layers with ReLU activation functions and one output layer with a sigmoid activation function, which is suitable for binary classification tasks.

## Part 3 - Training the ANN
This section covers the training of the ANN using the prepared training data.

### Compiling the ANN
The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as a metric to define its training configuration.

### Training the ANN
The model is trained using the training data, and a log directory is set up for TensorBoard visualization, enabling in-depth monitoring of the training process.

## Part 4 - Making Predictions and Evaluating the Model
In this section, the trained ANN model is used to make predictions and evaluate its performance.

### Predicting the Result of a Single Observation
A sample observation is provided to the model to predict the probability of churn, showcasing the model's ability to make individual predictions.

### Predicting the Test Set Results
The model is utilized to predict outcomes on the test set, and predicted probabilities are subsequently converted into binary predictions for practical interpretation.

### Making the Confusion Matrix
A confusion matrix and accuracy score are calculated to quantitatively assess the model's performance. Furthermore, a comprehensive classification report is provided, offering insights into precision, recall, F1-score, and support for both classes (0 and 1).

This code repository serves as a comprehensive example of building, training, and evaluating an Artificial Neural Network (ANN) for binary classification tasks using TensorFlow. It demonstrates various essential steps in the machine learning workflow and offers valuable insights into the model's performance.
