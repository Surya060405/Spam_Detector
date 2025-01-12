import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import string
import joblib

# Save the trained model


# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset (replace with your actual dataset path)
data = pd.read_csv("Spam_SMS.csv")
data = data.rename(columns={"Class": "label", "Message": "text"})

# Encode labels
data['spam'] = data['label'].map({'spam': 1, 'ham': 0}).astype(int)

# Preprocessing: Remove punctuation and stopwords
def remove_punctuation_and_stopwords(sms):
    sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
    sms_no_punctuation = "".join(sms_no_punctuation).split()
    sms_cleaned = [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]
    return sms_cleaned

# Tokenize and split data
data['text'] = data['text'].apply(remove_punctuation_and_stopwords)
X = data['text'].apply(lambda x: ' '.join(x))  # Convert list of words back to string
y = data['spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Define pipeline
pipe_MNB = Pipeline([
    ('tfidf_vec', TfidfVectorizer(analyzer=remove_punctuation_and_stopwords)),
    ('clf_MNB', MultinomialNB())
])

# Train model
pipe_MNB.fit(X_train, y_train)

# Save the trained model to a .pkl file


print("Model saved as model.pkl")
joblib.dump(pipe_MNB, 'model.pkl')
