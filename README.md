# Movie Review Sentiment Analysis

## ✨ Project Overview

This project performs sentiment analysis on a dataset of IMDb movie reviews. The goal is to classify movie reviews as either positive or negative based on their text content. It utilizes TF-IDF for feature extraction and various machine learning models from scikit-learn for classification.

## 🚀 Features

* **Data Loading & Preprocessing:** Reads movie review data from a CSV file using Pandas.
* **Text Cleaning:** Utilizes NLTK for tokenization, stop word removal, and potentially stemming/lemmatization.
* **Feature Extraction:** Employs TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
* **Sentiment Classification:** Trains and evaluates machine learning models (e.g., Naive Bayes, SVM, Logistic Regression from scikit-learn) to predict sentiment.
* **Evaluation:** Reports classification metrics (accuracy, precision, recall, F1-score).

## 📊 Dataset

The project uses a CSV file containing IMDb movie reviews. Each entry typically includes the review text and a corresponding sentiment label (e.g., 'positive', 'negative').

* **Expected File:** `imdb_reviews.csv` (or similar, you can specify the exact name in the instructions)
* **Format:** CSV with at least two columns: one for the review text and one for the sentiment label.

## 📁 File Structure
├── data/
│   └── imdb_reviews.csv  # Your dataset file
├── src/
│   └── sentiment_analyzer.py # Main Python script
├── requirements.txt      # List of dependencies
└── README.md             # This file
