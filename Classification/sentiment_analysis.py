from textblob import TextBlob

def analyze_sentiment(text):
    # Using TextBlob for sentiment analysis (positive, negative, neutral)
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)
    
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"
