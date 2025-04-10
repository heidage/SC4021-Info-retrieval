import spacy

def main():
# Load the pre-trained SpaCy model
    nlp = spacy.load("en_core_web_lg")

    # Example texts
    texts = [
        "Tesla is great!",
        "Bitcoin's price has dropped significantly recently.",
        "The stock market is uncertain due to economic changes."
    ]

    # Extract entities from the texts
    for text in texts:
        doc = nlp(text)
        print(f"Text: {text}")
        for ent in doc.ents:
            print(f"Entity: {ent.text}, Label: {ent.label_}")
        print()

if __name__ == "__main__":
    main()