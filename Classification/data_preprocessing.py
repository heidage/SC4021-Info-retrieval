from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to clean the text (remove stopwords, punctuation, etc.)
lemmatizer = WordNetLemmatizer()
# Function to clean text: Tokenization, Lemmatization, and Stopword removal
def clean_text(text):
    if isinstance(text, str):  # Check if the text is a string
        text = text.lower()  # Convert to lowercase
        tokens = word_tokenize(text)  # Tokenize the text

        # Lemmatize each token
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Additional preprocessing (like removing stopwords)
        stop_words = set(stopwords.words('english'))
        tokens_without_stopwords = [token for token in lemmatized_tokens if token not in stop_words]

        # Rejoin tokens into a single string before passing to TextBlob
        cleaned_text = " ".join(tokens_without_stopwords)
        return cleaned_text
    else:
        # If the text is not a string (e.g., NaN or float), return an empty string or a default value
        return ""  # Or return a string like 'Invalid Text' if needed
