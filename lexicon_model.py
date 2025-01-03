import sys
sys.path.append(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis')
from src.models.custom_lexicon import get_sentiment, get_sentiment_score

class LexiconModel:                 #object oriented wrapper class, that calls the original classes 
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        return get_sentiment(text)

    def get_sentiment_score(self, text):
        return get_sentiment_score(text)

if __name__ == "__main__":
    lexicon_model = LexiconModel()
    
    test_texts = [          #sample inputs to test, validate, and demonstrate the functionality of the sentiment analysis model.
        "The company's profits soared, exceeding all expectations.",
        "The stock market crashed, wiping out billions in value.",
        "The financial report showed mixed results for the quarter."
    ]
    
    for text in test_texts:
        sentiment = lexicon_model.analyze_sentiment(text)
        score = lexicon_model.get_sentiment_score(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Score: {score}\n")