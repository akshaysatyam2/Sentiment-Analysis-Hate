# Sentiment Analysis Model with Django

## Introduction
This project focuses on a Django-based Sentiment Analysis model that categorizes text sentiment as Positive, Negative, and Neutral.

## Project Components

### Model File
- The 'Model File' directory hosts the Sentiment Analysis model and the associated dataset.
- Text preprocessing is performed using libraries such as re, stopwords, and porter stemmer, along with one-hot encoding for labels.
- TensorFlow is employed for designing the model.
- This directory also includes a dataset file that serves as the training data for the model.

**Reason for Using ANN:**
The choice of Artificial Neural Networks (ANN) was motivated by the complexity of the data, which required robust learning and classification.

### SentimentAnalysis
- The 'SentimentAnalysis' directory encompasses a fully functional Django web application that implements the Sentiment Analysis model.
- This web application features two interactive pages: an input page and a result page.

## Data Preprocessing Steps

The provided code appears to be a text preprocessing and sentiment classification task using a neural network (likely an Artificial Neural Network or ANN). Here's a summary of the text preprocessing steps:

1. Input Text: Data from the dataset.
2. Regular Expression Substitution: 're.sub('[^a-zA-Z]', ' ', new_review)' is used to remove all characters except alphabetic letters. This step ensures that only letters remain in the text.
3. Lowercasing: 'new_review = new_review.lower()' converts all letters in the text to lowercase. This step is typically done to ensure uniformity in the text data.
4. Tokenization: 'new_review = new_review.split()' splits the text into a list of words. This is essential for further processing as it breaks down the text into individual words.
5. Stemming: The code initializes a Porter Stemmer ('ps') and applies it to each word in the tokenized text. Stemming reduces words to their root forms, which helps in simplifying the vocabulary. Example: "best" becomes "best."
6. Stopword Removal: It removes common English stopwords (e.g., "and," "the," "is") using the NLTK library. However, it retains the word "not" since it is crucial for sentiment analysis.
7. Joining Tokens: The processed tokens are joined back together into a single string. This step is often necessary before applying machine learning models.
8. Text Vectorization: 'cv.transform(new_corpus).toarray()' converts the preprocessed text into a numerical format that can be used as input for a machine learning model. It's likely that 'cv' represents a CountVectorizer or a similar text vectorization method.

## Model Architecture and Parameters

- The neural network is constructed as a sequential model.
- Four hidden layers are included, each with 32 units and ReLU activation functions.
- The output layer consists of three units with softmax activation for multi-class classification.

### Training Parameters
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metric for Evaluation: Accuracy

### Training Process
- Training data: X_train and y_train
- Validation split: 20%
- Batch size: 32
- Maximum epochs: 100
- Early stopping is applied with a patience of 10 to prevent overfitting.

## Evaluation Results and Analysis

### Final Evaluation Metrics
- Confusion Matrix:
[[3207 0 1206]
[ 5 0 1]
[1625 0 2185]]
- Overall Accuracy: 0.6552

### Training Analysis
- The training process ran for a total of 11 epochs.
- The model achieved a peak validation accuracy of 0.6672 around epoch 2.
- After epoch 2, there is a gradual decrease in validation accuracy, which might indicate overfitting.
- The final confusion matrix shows the distribution of true positive, true negative, and false positive predictions for each class.
- The overall accuracy of the model is approximately 65.52%.

**Note:** There was incorrect data in the dataset which is also responsible for lower accuracy.

## Instructions for Working with the Project

### Software Requirements
Ensure you have the following Python libraries and dependencies installed:
1. Stopwords
2. Django
3. NumPy
4. Pandas
5. Pickle
6. Bootstrap
7. TensorFlow
8. PyTorch

### Setup Instructions

1. Model File:
 - Import the model file and execute all its components.

2. Django Project:
 a) Clone the project repository from 'https://github.com/akshaysatyam2/Sentiment-Analysis-Hate'.
    You can use the following command:
    ```
    git clone https://github.com/akshaysatyam2/Sentiment-Analysis-Hate
    ```
 b) Optionally, make migrations using the following commands:
    ```
    python manage.py makemigrations
    python manage.py migrate
    ```
 c) Run the Django development server:
    ```
    python manage.py runserver
    ```

**Note:** If you encounter any errors during setup or usage, feel free to seek assistance or search for solutions online.

