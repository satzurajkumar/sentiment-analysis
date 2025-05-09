import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Dataset ---
try:
    # Update this path if your file is located elsewhere
    df = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found. Please download it from Kaggle and place it in the correct directory.")
    exit()

print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
df.info()
print("\n--- Sentiment Distribution ---")
print(df['sentiment'].value_counts())

# --- 2. Preprocess the Text Data ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    return " ".join(cleaned_tokens)

print("\n--- Preprocessing a sample review ---")
sample_review_original = df['review'][0]
sample_review_processed = preprocess_text(sample_review_original)
print("Original:", sample_review_original[:200] + "...") # Print first 200 chars
print("Processed:", sample_review_processed[:200] + "...")

# Apply preprocessing to the 'review' column
# This might take a few minutes for 50k reviews
print("\n--- Starting text preprocessing for all reviews (this may take a while)... ---")
df['processed_review'] = df['review'].apply(preprocess_text)
print("--- Text preprocessing complete. ---")

print("\n--- Processed Dataset Head ---")
print(df.head())

# --- 3. Feature Extraction (TF-IDF) ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features

print("\n--- Starting TF-IDF vectorization... ---")
X = tfidf_vectorizer.fit_transform(df['processed_review'])
# Convert sentiment labels to numerical (positive: 1, negative: 0)
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
print("--- TF-IDF vectorization complete. ---")
print("Shape of TF-IDF matrix:", X.shape)

# --- 4. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n--- Data Splitting ---")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# --- 5. Train Sentiment Analysis Models ---

# Model 1: Multinomial Naive Bayes
print("\n--- Training Multinomial Naive Bayes model... ---")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("--- Multinomial Naive Bayes training complete. ---")

# Model 2: Logistic Regression
print("\n--- Training Logistic Regression model... ---")
lr_model = LogisticRegression(solver='liblinear', random_state=42) # liblinear is good for smaller datasets
lr_model.fit(X_train, y_train)
print("--- Logistic Regression training complete. ---")


# --- 6. Evaluate the Models ---

# Evaluate Naive Bayes
print("\n--- Multinomial Naive Bayes Evaluation ---")
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy: {accuracy_nb:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6,4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Evaluate Logistic Regression
print("\n--- Logistic Regression Evaluation ---")
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- 7. Make Predictions on New Reviews (Example) ---
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    probability = model.predict_proba(vectorized_text)
    if prediction[0] == 1:
        return "Positive", probability[0][1]
    else:
        return "Negative", probability[0][0]

print("\n--- Making predictions on new reviews (using Logistic Regression model as it's often slightly better) ---")
new_review_1 = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
sentiment_1, prob_1 = predict_sentiment(new_review_1, lr_model, tfidf_vectorizer)
print(f"Review: \"{new_review_1}\" \nSentiment: {sentiment_1} (Confidence: {prob_1:.4f})\n")

new_review_2 = "A complete waste of time. The story was boring and the characters were flat."
sentiment_2, prob_2 = predict_sentiment(new_review_2, lr_model, tfidf_vectorizer)
print(f"Review: \"{new_review_2}\" \nSentiment: {sentiment_2} (Confidence: {prob_2:.4f})\n")

new_review_3 = "It was an okay movie, not great but not terrible either. Some good moments."
# This kind of neutral review is harder to classify with a binary model.
# The model will pick the class it leans towards more.
sentiment_3, prob_3 = predict_sentiment(new_review_3, lr_model, tfidf_vectorizer)
print(f"Review: \"{new_review_3}\" \nSentiment: {sentiment_3} (Confidence: {prob_3:.4f})\n")