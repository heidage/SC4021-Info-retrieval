import time
from data_preprocessing import clean_text
from subjectivity_detection import detect_subjectivity
from sentiment_analysis import analyze_sentiment
from named_entity_recognition import extract_entities
from model import train_model, predict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer

if __name__ == "__main__":

    # Sample data (Reddit posts about stocks)
    texts = [
        "Tesla stock is performing so much more better than expected!",
        "Facebook stock has dropped significantly recently.",
        "The stock market is uncertain due to economic changes."
    ]
    
    # Sample labels for training (these would be your actual sentiment labels in a real dataset)
    labels = ["positive", "negative", "neutral"]

    # --- 1. Preprocess the text ---
    start_time = time.time()
    cleaned_texts = [clean_text(text) for text in texts]
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing Time: {preprocessing_time:.4f} seconds")
    
    # --- 2. Detect subjectivity ---
    start_time = time.time()
    subjectivity = [detect_subjectivity(text) for text in cleaned_texts]
    subjectivity_time = time.time() - start_time
    print(f"Subjectivity Detection Time: {subjectivity_time:.4f} seconds")

    # --- 3. Analyze sentiment ---
    start_time = time.time()
    sentiment = [analyze_sentiment(text) for text in cleaned_texts]
    sentiment_analysis_time = time.time() - start_time
    print(f"Sentiment Analysis Time: {sentiment_analysis_time:.4f} seconds")
    
    # --- 4. Extract named entities ---
    start_time = time.time()
    entities = [extract_entities(text) for text in cleaned_texts]
    entity_extraction_time = time.time() - start_time
    print(f"Entity Extraction Time: {entity_extraction_time:.4f} seconds")

    # --- 5. Train the "model" ---
    start_time = time.time()
    model, vectorizer = train_model(texts, labels)
    training_time = time.time() - start_time
    print(f"Model Training Time: {training_time:.4f} seconds")

    # --- 6. Split the data into train and test sets ---
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=42)
    
    # --- 7. Make predictions on the test set ---
    start_time = time.time()
    y_pred = predict(model, vectorizer, X_test)
    prediction_time = time.time() - start_time
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    
    # --- 8. Print classification report ---
    print(classification_report(y_test, y_pred))

    # --- 9. Sentiment scores using VADER ---
    sid = SentimentIntensityAnalyzer() 
    for text in texts:
        score = sid.polarity_scores(text)
        print(f"Text: {text}")
        print(f"Sentiment Scores: {score}")
        print("Compound Score:", score['compound'])

    # --- Performance Metrics: Records Classified Per Second ---
    total_time = preprocessing_time + subjectivity_time + sentiment_analysis_time + entity_extraction_time + training_time + prediction_time
    records_per_second = len(texts) / total_time  # records classified per second

    print(f"\n--- Performance Metrics ---")
    print(f"Total Time for Preprocessing, Classification, and Prediction: {total_time:.4f} seconds")
    print(f"Records Classified per Second: {records_per_second:.2f} records/second")
    
    # --- Scalability Discussion ---
    print("\n--- Scalability Discussion ---")
    print(f"{len(texts)} records processed in {total_time:.4f} seconds")
