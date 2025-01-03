import sys
sys.path.append(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VADERModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores['compound']

        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def get_sentiment_score(self, text):
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

if __name__ == "__main__":
    # Test the VADER model
    vader_model = VADERModel()
    
    test_texts = [
        "The company's profits soared, exceeding all expectations.",
        "The stock market crashed, wiping out billions in value.",
        "The financial report showed mixed results for the quarter."
    ]
    
    for text in test_texts:
        sentiment = vader_model.analyze_sentiment(text)
        score = vader_model.get_sentiment_score(text)
        print(f"Text: {text}")
        print(f"VADER Sentiment: {sentiment}")
        print(f"VADER Score: {score}\n")