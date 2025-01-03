import sys
sys.path.append(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis')
from src.models.lexicon_model import LexiconModel
from src.models.ml_model import MLModel

class HybridSentimentAnalyzer:                          #creating hybrid ML model
    def __init__(self, ml_model_file):
        self.lexicon_model = LexiconModel()
        self.ml_model = MLModel()
        self.ml_model.load_model(ml_model_file)

    def analyze_sentiment(self, text):
        lexicon_sentiment = self.lexicon_model.analyze_sentiment(text)
        ml_sentiment = self.ml_model.predict(text)
        
        if lexicon_sentiment == ml_sentiment:
            return lexicon_sentiment
        else:
            # If there's a disagreement, use the sentiment score from the lexicon model as a tiebreaker
            sentiment_score = self.lexicon_model.get_sentiment_score(text)
            return 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'

    def get_sentiment_score(self, text):
        return self.lexicon_model.get_sentiment_score(text)

if __name__ == "__main__":
    analyzer = HybridSentimentAnalyzer(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis\data\ml_model.joblib')
    
    test_texts = [
        "The company's profits soared, exceeding all expectations.",
        "The stock market crashed, wiping out billions in value.",
        "The financial report showed mixed results for the quarter."
    ]
    
    for text in test_texts:
        sentiment = analyzer.analyze_sentiment(text)
        score = analyzer.get_sentiment_score(text)
        print(f"Text: {text}")
        print(f"Hybrid Sentiment: {sentiment}")
        print(f"Sentiment Score: {score}\n")