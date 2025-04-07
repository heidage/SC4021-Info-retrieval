from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

# Function to compute sentiment using VADER
def compute_sentiment(texts):
    # Initialize the SentimentIntensityAnalyzer from VADER
    sid = SentimentIntensityAnalyzer()
    
    # List to store sentiment labels (positive, negative, neutral)
    sentiment_labels = []
    
    # Loop through each text and compute the sentiment
    for text in texts:
        sentiment_score = sid.polarity_scores(text)['compound']
        
        # Assign sentiment labels based on the compound score
        if sentiment_score > 0:
            sentiment_labels.append('positive')
        elif sentiment_score < 0:
            sentiment_labels.append('negative')
        else:
            sentiment_labels.append('neutral')
    
    return sentiment_labels

# Function to train the model (Note: sentiment analysis is rule-based, no model training required)
def train_model(X_train, y_train=None):
    # Compute sentiment labels using VADER lexicon
    y_train_pred = compute_sentiment(X_train)
    
    # Return the "model" (which in this case is just the VADER lexicon)
    return None, None  # No need for a traditional model

# Function to make predictions (same as above, using VADER)
def predict(model, vectorizer, X_test):
    # Compute sentiment for the test set using VADER lexicon
    y_test_pred = compute_sentiment(X_test)
    
    return y_test_pred

# Optional: Save the "model" (in this case, no need for saving, but you can save other results)
def save_model(model, vectorizer, filename="sentiment_model.pkl"):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump((model, vectorizer), f)

# Optional: Load the model (again, not needed as we don't have a traditional model)
def load_model(filename="sentiment_model.pkl"):
    import pickle
    with open(filename, 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer