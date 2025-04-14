from sklearn.feature_extraction.text import CountVectorizer

def extract_aspects(texts):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))  # Extract unigrams and bigrams
    aspects = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out()
