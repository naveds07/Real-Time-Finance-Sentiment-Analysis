# src/models/ml_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

import sys
sys.path.append(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis')

from src.data_preprocessing.financial_phrasebank_loader import load_financial_phrasebank

class MLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)               #text vectorization
        self.classifier = MultinomialNB()

    def train(self, X, y):                                                #splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vectorized, y_train)
        
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vectorized)
        
        print(classification_report(y_test, y_pred))

    def predict(self, text):
        vectorized_text = self.vectorizer.transform([text])
        return self.classifier.predict(vectorized_text)[0]

    def save_model(self, filename):
        joblib.dump((self.vectorizer, self.classifier), filename)

    def load_model(self, filename):
        self.vectorizer, self.classifier = joblib.load(filename)

if __name__ == "__main__":
    # Loading Financial PhraseBank dataset
    df = load_financial_phrasebank(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis\data\financial_phrasebank.txt')
    
    ml_model = MLModel()
    ml_model.train(df['text'], df['sentiment'])
    
    # Saving the trained model
    ml_model.save_model(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis\data\ml_model.joblib')
    
    # Testing the ML model
    test_texts = [
        "The company's profits soared, exceeding all expectations.",
        "The stock market crashed, wiping out billions in value.",
        "The financial report showed mixed results for the quarter."
    ]
    
    for text in test_texts:
        sentiment = ml_model.predict(text)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment}\n")