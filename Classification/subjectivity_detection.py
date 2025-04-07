from textblob import TextBlob

def detect_subjectivity(text):
    # Use TextBlob to calculate subjectivity
    blob = TextBlob(text)
    return blob.sentiment.subjectivity  # Returns a value between 0 and 1, where 1 is very subjective
