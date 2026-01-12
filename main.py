import nltk
import re
import string
# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

#Input text
text = ("Hi my name is Trung Tien, I like Gacha game such as Honkai Star Rail, Fate Grand Oder, Blue Archive."
        "I also love reading light novel and watching anime during my free time."
        )
print("Original Text:\n", text)
print("-" * 60)

# Tokenization
tokens = word_tokenize(text)
print("Tokens:\n", tokens)
print("-" * 60)

# Lowercasing
lower_tokens = [token.lower() for token in tokens]
print("Lowercased Tokens:\n", lower_tokens)
print("-" * 60)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in lower_tokens if token not in stop_words]
print("Tokens after Stopword Removal:\n", filtered_tokens)
print("-" * 60)

# Punctuation Removal
punctuation_table = str.maketrans('', '', string.punctuation)
punctuation_free_tokens = [token.translate(punctuation_table) for token in filtered_tokens if token.translate(punctuation_table)]
print("Punctuation-free Tokens:\n", punctuation_free_tokens)
print("-" * 60)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in punctuation_free_tokens]
print("Stemmed Tokens:\n", stemmed_tokens)
print("-" * 60)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in punctuation_free_tokens]
print("Lemmatized Tokens:\n", lemmatized_tokens)
print("-" * 60)

# Text Normalization Function
def normalize_text(input_text):
    tokens = word_tokenize(input_text)
    lower_tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lower_tokens if token not in stop_words]
    punctuation_table = str.maketrans('', '', string.punctuation)
    punctuation_free_tokens = [token.translate(punctuation_table) for token in filtered_tokens if token.translate(punctuation_table)]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in punctuation_free_tokens]
    return lemmatized_tokens
