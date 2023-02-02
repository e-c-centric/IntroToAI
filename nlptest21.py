import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
nltk.download('movie_reviews')

# Load movie reviews corpus
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# Split data into training and testing sets
training_reviews = reviews[:1900]
testing_reviews = reviews[1900:]

# Define feature extractor
def extract_features(words):
    return dict([(word, True) for word in words])

# Train NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(training_reviews)

# Evaluate classifier accuracy on testing data
accuracy = accuracy(classifier, testing_reviews)
print("Accuracy: ", accuracy)

# Use classifier to classify new text
text = "I love this movie, it's so good and enjoyable."
features = extract_features(text.split())
sentiment = classifier.classify(features)
print("Sentiment: ", sentiment)
