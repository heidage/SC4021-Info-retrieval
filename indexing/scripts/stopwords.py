from nltk.corpus import stopwords
import nltk


# Download the stopwords corpus if not already downloaded
print("Downloading stopwords corpus...")
nltk.download('stopwords')
print("Dowloaded stopwords corpus")

# save stopwords to a text file
with open("../build/solr_config/stopwords.txt", "w") as file:
    for word in sorted(stopwords.words('english')):
        file.write(word + "\n")

print("Stopwords saved to ../build/solr_config/stopwords.txt")