import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Example text
text = "f"

# Get sentiment score
score = sia.polarity_scores(text)["compound"]

# Determine sentiment based on score
if score > 0:
    print("Positive sentiment")
elif score < 0:
    print("Negative sentiment")
else:
    print("Neutral sentiment")