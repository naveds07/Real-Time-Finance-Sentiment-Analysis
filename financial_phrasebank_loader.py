import pandas as pd

def load_financial_phrasebank(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Changed encoding to handle encoding issues
        lines = file.readlines()
    
    data = []
    for line in lines:
        sentence, sentiment = line.strip().split('@')
        data.append({'text': sentence, 'sentiment': sentiment})
    
    df = pd.DataFrame(data)
    df['sentiment'] = df['sentiment'].map({'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'})
    return df

if __name__ == "__main__":
    # Test the loader
    df = load_financial_phrasebank('C:/Users/Naved/OneDrive/Desktop/Sentiment Analysis/Sentiment Analysis Project/Sentiment Analysis Project/financial_sentiment_analysis/data')
    print(df.head())
    print(df['sentiment'].value_counts())
