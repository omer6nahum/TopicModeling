import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer


def calc_ctfidf(sentences, labels):
    # calculating tfidf, where each "document" is a concatenation of all sentences from the same class
    labels_set = list(set(labels))
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    texts = []
    for label in labels_set:
        rel_sentences = np.array(sentences)[np.array(labels) == label]
        texts.append(' '.join(rel_sentences))

    matrix = vectorizer.fit_transform(texts)
    # return: words list, ctfidf matrix, labels list
    return vectorizer.get_feature_names_out(), np.array(matrix.todense()), labels_set


def common_words(sentences, labels, k=10):
    assert len(sentences) == len(labels)
    words_list, ctfidf, labels_set = calc_ctfidf(sentences, labels)
    most_common = dict()

    for i, label in enumerate(labels_set):
        k_common_indices = np.argsort(ctfidf[i])[-k:][::-1]
        k_common_words_scores = dict(zip(words_list[k_common_indices], ctfidf[i, k_common_indices]))
        most_common[label] = k_common_words_scores

    return most_common


def word_cloud(k_common_words):
    cloud = WordCloud(background_color="white", width=500, height=150, colormap='copper')\
        .generate_from_frequencies(k_common_words)
    plt.figure(figsize=(11, 7))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def topic_labeling(common_words, model):
    topic_labels = dict()
    for label, words_dict in common_words.items():
        words = list(words_dict.keys())
        other_words = list(itertools.chain(*[list(other_words_dict.keys())
                                             for other_label, other_words_dict in common_words.items()
                                             if other_label != label]))
        # ignoring words not in vocabulary for gensim model
        words = [w for w in words if w in model.key_to_index]
        other_words = [w for w in other_words if w in model.key_to_index]

        # get most similar word by embeddings similarity to words, and dissimilarity to other_words
        topic_labels[label] = model.most_similar(positive=list(zip(words, [+1.0]*len(words))),
                                                 # negative=list(zip(other_words, [-0.3]*len(other_words))),
                                                 topn=5)
    return topic_labels