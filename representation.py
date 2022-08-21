import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from umap import UMAP
from gensim.downloader import load as gensim_load
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans


class SentEmbeddings(ABC):
    @abstractmethod
    def embed(self, sentences):
        pass


class SentBERT(SentEmbeddings):
    # all model names: https://www.sbert.net/docs/pretrained_models.html
    def __init__(self, model_name='all-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, sentences):
        return self.model.encode(sentences)


class WEClustering(SentEmbeddings):
    # todo: currently not working well. maybe there is a bug.
    def __init__(self, k):
        self.clustering_model = KMeans(n_clusters=k, random_state=2)
        self.embd_model = WordEmbeddings()

    def embed(self, sentences):
        # sentences is a list of sentences,
        # --X-- each sentence is represented by list of tokens(words).

        # tfidf
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        sent_term_matrix = vectorizer.fit_transform(sentences)  # (n_sentences, n_terms)
        vocab = sorted(vectorizer.vocabulary_.keys(), key=lambda x: vectorizer.vocabulary_[x])

        # clustering of words
        X = self.embd_model.embed(vocab)  # possibility to reduce dimensions
        self.clustering_model.fit(X)
        word_clusters = self.clustering_model.labels_
        word_clusters_onehot = OneHotEncoder().fit_transform(np.expand_dims(word_clusters, 1))

        # constructing new representation for each sentence
        sent_concept_matrix = sent_term_matrix.todense() @ word_clusters_onehot.todense()
        # sent_concept_matrix = sent_term_matrix.multiply(word_clusters_onehot).todense()
        return sent_concept_matrix


class WordEmbeddings:
    def __init__(self, model_name='word2vec-google-news-300'):
        # self.model = gensim.models.Word2Vec(text, min_count=1, vector_size=100, window=5, sg=1)
        self.model = gensim_load(model_name)

    def embed(self, words):
        embeddings = []
        for w in words:
            try:
                embeddings.append(self.model[w])
            except KeyError:
                embeddings.append(np.ones(300) / 300)
        return np.array(embeddings)


class Reducer:
    @abstractmethod
    def fit(self, X):
        pass

    def transform(self, X):
        pass


class Umap(Reducer):
    def __init__(self, n_components, n_init_observations):
        init = 'spectral'
        if n_init_observations <= n_components + 1:
            init = 'random'
        self.reducer = UMAP(n_components=n_components, n_neighbors=5, random_state=2, init=init)

    def fit(self, X):
        self.reducer = self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)