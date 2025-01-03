import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')                                  #Punkt tokenizer from NLTK, for tokenizing sentences into words.

class UnsupervisedModel:
    def __init__(self, n_clusters=3):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.clustering = KMeans(n_clusters=n_clusters, random_state=42)        
        self.word2vec = None

    def train(self, texts):
        # TF-IDF vectorization
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Word2Vec training
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        self.word2vec = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        
        # Combining TF-IDF and Word2Vec features
        word2vec_matrix = np.array([self.get_text_vector(text) for text in texts])
        combined_features = np.hstack((tfidf_matrix.toarray(), word2vec_matrix))
        
        # Performing clustering
        self.clustering.fit(combined_features)
        
        # Analyzing clusters
        self.analyze_clusters(texts)

    def get_text_vector(self, text):
        words = word_tokenize(text.lower())
        word_vectors = [self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.word2vec.vector_size)

    def analyze_clusters(self, texts):
        cluster_texts = [[] for _ in range(self.clustering.n_clusters)]
        for text, label in zip(texts, self.clustering.labels_):
            cluster_texts[label].append(text)
        
        for i, texts in enumerate(cluster_texts):
            print(f"Cluster {i}:")
            print(f"Size: {len(texts)}")
            print("Top terms:")
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf_matrix = vectorizer.fit_transform(texts)
            top_terms = vectorizer.get_feature_names_out()[np.argsort(tfidf_matrix.toarray().sum(axis=0))[-10:]]
            print(", ".join(top_terms))
            print()

    def predict(self, text):
        tfidf_vector = self.vectorizer.transform([text]).toarray()
        word2vec_vector = self.get_text_vector(text).reshape(1, -1)
        combined_vector = np.hstack((tfidf_vector, word2vec_vector))
        cluster = self.clustering.predict(combined_vector)[0]
        return f"Cluster {cluster}"

if __name__ == "__main__":
    # Test on unsupervised model
    texts = [
        "The company's profits soared, exceeding all expectations.",
        "The stock market crashed, wiping out billions in value.",
        "The financial report showed mixed results for the quarter.",
        "Investors are optimistic about the company's future prospects.",
        "The company announced a significant layoff due to restructuring.",
    ]
    
    unsupervised_model = UnsupervisedModel()
    unsupervised_model.train(texts)
    
    for text in texts:
        cluster = unsupervised_model.predict(text)
        print(f"Text: {text}")
        print(f"Assigned Cluster: {cluster}\n")