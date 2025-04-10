import spacy

nlp = spacy.load("en_core_web_lg")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extracting company names (organizations)
    return entities
