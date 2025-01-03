FINANCIAL_LEXICON = {
    'positive': [
        'growth', 'profits', 'success', 'increase', 'gain', 'uptrend', 'bullish',
        'outperform', 'exceed', 'beat', 'strong', 'soared', 'positive', 'upgrade',
        'opportunity', 'innovation', 'efficient', 'recovery', 'expansion', 'dividend'
    ],
    'negative': [
        'loss', 'decline', 'debt', 'decrease', 'downtrend', 'bearish', 'underperform',
        'miss', 'weak', 'poor', 'negative', 'downgrade', 'risk', 'bankruptcy', 'layoff',
        'recession', 'default', 'litigation', 'volatility', 'uncertainty', 'crashed'
    ],
    'neutral': [
        'steady', 'stable', 'unchanged', 'maintain', 'hold', 'flat', 'mixed',
        'balanced', 'moderate', 'average', 'in-line', 'expected', 'forecast',
        'estimate', 'guidance', 'outlook', 'projection', 'target', 'plan', 'strategy'
    ]
}

def get_sentiment_score(text):                          #counts number of positive and negative words
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in FINANCIAL_LEXICON['positive'])
    negative_count = sum(1 for word in words if word in FINANCIAL_LEXICON['negative'])
    
    if positive_count > negative_count:                 #compares numbers of positive and negative words    
        return 1  # Positive
    elif negative_count > positive_count:
        return -1  # Negative
    else:
        return 0  # Neutral

def get_sentiment(text):
    score = get_sentiment_score(text)
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'