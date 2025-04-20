import spacy
from nltk.tokenize import word_tokenize
import nltk
from nltk import pos_tag

nlp = spacy.load("en_core_web_lg")

def extract_entities(text):
    # Step 1: Use spaCy to extract entities (like companies, products, etc.)
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]  # Extracting organizations

    # Step 2: Apply POS tagging to the tokens in the text
    tokens = word_tokenize(text)  # Tokenize the text
    pos_tags = pos_tag(tokens)  # Get POS tags for the tokens

    # Step 3: Extract proper nouns (NNP) or important nouns (NN) using POS tagging
    nouns = [word for word, tag in pos_tags if tag in ["NNP","NN"]]  # Extracting proper nouns and common nouns

    # Combine NER and POS results
    refined_entities = list(set(entities + nouns))  # Combine and remove duplicates
    return refined_entities