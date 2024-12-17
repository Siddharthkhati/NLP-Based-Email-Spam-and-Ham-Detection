import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

def train_model():
    # Load dataset
    data = pd.read_csv('dataset.csv') 
    X = data[['subject', 'message']]
    X['text'] = X['subject'] + ' ' + X['message']
    X['processed_text'] = X['text'].apply(preprocess_text)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X['processed_text'], y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)  
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vect, y_train)

    return clf, vectorizer  

def predict_text(clf, vectorizer, text):
    """
    Predicts whether the given text is spam or ham.

    Parameters:
    - clf: The trained classifier model.
    - vectorizer: The fitted TfidfVectorizer object.
    - text: The text to be classified.

    Returns:
    - prediction: The predicted class label.
    """
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = clf.predict(vectorized_text)
    return prediction[0]

