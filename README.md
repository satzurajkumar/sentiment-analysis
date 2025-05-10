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
```bash
├── data/
│   └── imdb_reviews.csv  # Your dataset file
├── src/
│   └── sentiment_analyzer.py # Main Python script
├── requirements.txt      # List of dependencies
└── README.md             # This file
```

## 🛠️ Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    * On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK data:**
    You might need to download necessary NLTK data like 'punkt' and 'stopwords'. Run Python and use the NLTK downloader:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # You might need other data depending on your preprocessing steps
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')
    ```

## 🏃 How to Run

1.  Make sure your dataset file (`imdb_reviews.csv`) is placed in the `data/` directory.
2.  Run the main Python script:

    ```bash
    python src/sentiment_analyzer.py
    ```

    The script will load the data, preprocess it, train the model, evaluate it, and print the results.

## 📦 Dependencies

The project relies on the following Python libraries, listed in `requirements.txt`:

* `pandas`
* `scikit-learn`
* `nltk`
