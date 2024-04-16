import nltk
from nltk.corpus import movie_reviews
import random

# ---------------------------------------------------------------------------------------
# Load movie reviews dataset
nltk.download('movie_reviews')
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# ---------------------------------------------------------------------------------------
# Shuffle the documents
random.shuffle(documents)

# ---------------------------------------------------------------------------------------
# Define a feature extractor
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:2000]

# ---------------------------------------------------------------------------------------
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# ---------------------------------------------------------------------------------------
# Extract features and split into training and testing datasets
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

# ---------------------------------------------------------------------------------------
# Train a Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# ---------------------------------------------------------------------------------------
# Evaluate classifier
print("Accuracy:", nltk.classify.accuracy(classifier, test_set))

# ---------------------------------------------------------------------------------------
# Example usage
review = "This movie was fantastic! I loved every moment of it."
features = document_features(review.split())
print("Sentiment:", classifier.classify(features))
