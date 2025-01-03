import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import sys
sys.path.append(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis')


from src.models.lexicon_model import LexiconModel

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the file at {file_path}. Make sure it's a valid CSV.")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Removing special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Converting to lowercase
    text = text.lower()
    return text

def tokenize_and_lemmatize(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        print("Error: NLTK stopwords not found. Please run the nltk_downloader.py script.")
        return ""

    lemmatizer = WordNetLemmatizer()
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        print("Error: NLTK punkt not found. Please run the nltk_downloader.py script.")
        return ""

    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def preprocess_data(df):
    if df is None or df.empty:
        print("Error: No data to preprocess")
        return None

    if 'webTitle' not in df.columns:
        print("Error: 'webTitle' column not found in the dataset")
        return None

    print("Cleaning text...")
    df['cleaned_text'] = df['webTitle'].apply(clean_text)
    
    print("Tokenizing and lemmatizing...")
    df['processed_text'] = df['cleaned_text'].apply(tokenize_and_lemmatize)
    
    print("Performing initial sentiment analysis...")
    lexicon_model = LexiconModel()
    df['sentiment'] = df['processed_text'].apply(lexicon_model.analyze_sentiment)
    
    return df

if __name__ == "__main__":
    file_path = r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis\data\financial_news_data.csv'
    df = load_data(file_path)
    if df is not None:
        preprocessed_df = preprocess_data(df)
        if preprocessed_df is not None:
            preprocessed_df.to_csv('C:/Users/Naved/OneDrive/Desktop/Sentiment Analysis/Sentiment Analysis Project/Sentiment Analysis Project/financial_sentiment_analysis/data/financial_news_data.csv', index=False)
            print("Preprocessing completed and saved to 'preprocessed_financial_news_data.csv'")