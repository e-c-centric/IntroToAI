# Import the TextBlob library
from textblob import TextBlob

# Define a sentence or text to analyze
text = "My name is Elikem!"

# Create a TextBlob object from the text
blob = TextBlob(text)

# Use the TextBlob object to get the sentiment polarity
polarity = blob.sentiment.polarity

# Check the sentiment polarity and print the result
if polarity > 0:
    print("Sentiment: Positive")
elif polarity == 0:
    print("Sentiment: Neutral")
else:
    print("Sentiment: Negative")

# Import the necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the input data and target output
input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
target_output = np.array([[0],[1],[1],[0]])

# Define the model architecture
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(input_data, target_output, epochs=100, batch_size=1)

# Test the model
test = np.array([[1,1]])
result = model.predict(test)
rounded = [round(x[0]) for x in result]
print("Result:", rounded)
