import time
import pandas as pd
from data_preprocessing import clean_text
from textblob import TextBlob
from named_entity_recognition import extract_entities  # Assuming you have this function ready
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer

# Function to load data from Excel
def load_data_from_excel(file_path):
    # Read the Excel file
    df = pd.read_csv(file_path)
    
    # Assuming the Excel file has columns 'text' for the comments
    texts = df['cleaned_body'].tolist()  # List of texts (Reddit posts)
    return df, texts

# Function to perform sentiment analysis with TextBlob
def analyze_sentiment_with_textblob(texts):
    subjectivity = []
    polarity = []
    for text in texts:
        blob = TextBlob(text)
        subjectivity.append(blob.sentiment.subjectivity)
        polarity.append(blob.sentiment.polarity)
    return subjectivity, polarity

# Main execution
if __name__ == "__main__":

    # Load data from Excel sheet
    file_path = "/Users/jaredog/Downloads/cleaned_stock_data (1).csv"  # Replace with your actual Excel file path
    df, texts = load_data_from_excel(file_path)

    # --- 1. Preprocess the text ---
    start_time = time.time()
    cleaned_texts = [clean_text(text) for text in texts]
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing Time: {preprocessing_time:.4f} seconds")

    # --- 2. Sentiment Analysis with TextBlob ---
    start_time = time.time()
    subjectivity, polarity = analyze_sentiment_with_textblob(cleaned_texts)
    sentiment_analysis_time = time.time() - start_time
    print(f"Sentiment Analysis Time: {sentiment_analysis_time:.4f} seconds")
    
    # Append the sentiment analysis results to the dataframe
    df['subjectivity'] = subjectivity
    df['polarity'] = polarity

    # Determine the overall sentiment (positive, negative, neutral)
    sentiment_labels = []
    for polarity_score in polarity:
        if polarity_score > 0:
            sentiment_labels.append('positive')
        elif polarity_score < 0:
            sentiment_labels.append('negative')
        else:
            sentiment_labels.append('neutral')
    
    df['sentiment'] = sentiment_labels

    # --- 3. Remove neutral sentiment rows ---
    df = df[df['sentiment'] != 'neutral']  # Drop all rows where sentiment is 'neutral'

    # --- 4. Named Entity Recognition ---
    start_time = time.time()
    # Apply NER only to the remaining (non-neutral) rows
    remaining_texts = df['cleaned_body'].tolist()  # Only the rows with non-neutral sentiment
    entities = [extract_entities(text) for text in remaining_texts]
    entity_extraction_time = time.time() - start_time
    print(f"Entity Extraction Time: {entity_extraction_time:.4f} seconds")

    # Append extracted entities to the dataframe
    df['entities'] = entities

    # --- 5. Save the modified data back to CSV ---
    output_file_path = "modified_comments_without_neutral.csv"
    df.to_csv(output_file_path, index=False)
    print(f"Modified data saved to {output_file_path}")

    # --- Performance Metrics ---
    total_time = preprocessing_time + sentiment_analysis_time + entity_extraction_time
    records_per_second = len(df) / total_time  # records classified per second

    print(f"\n--- Performance Metrics ---")
    print(f"Total Time for Preprocessing, Sentiment Analysis, and Entity Extraction: {total_time:.4f} seconds")
    print(f"Records Classified per Second: {records_per_second:.2f} records/second")
