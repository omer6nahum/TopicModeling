import re
import nltk
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger')

STOP_WORDS = set(stopwords.words("english"))
PUNCTUATIONS = list(string.punctuation)
LEMMA = WordNetLemmatizer()


def remove_punc(text, punc_list):
    for punc in punc_list:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()


def remove_stopwords(list_of_words, stopwords_list):
    return [w for w in list_of_words if w not in stopwords_list]


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_wikipedia_numbers(text):
    return re.sub(r'(\[\d+\])', '', text)


def get_wordnet_pos(tag):
    """
    from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


def lemmatization(list_of_words, lemma):
    word_pos_list = pos_tag(list_of_words)  # list of tuples (word, pos)
    return [lemma.lemmatize(w, get_wordnet_pos(p)) for w, p in word_pos_list]


def clean_sentence(text):
    text = text.lower()
    text = remove_wikipedia_numbers(text)
    text = remove_punc(text, PUNCTUATIONS)
    list_of_words = word_tokenize(text)
    list_of_words = remove_stopwords(list_of_words, STOP_WORDS)
    list_of_words = lemmatization(list_of_words, lemma=LEMMA)
    text = ' '.join(list_of_words)
    return text


def sentence_preprocess(text):
    sentences = sent_tokenize(text)
    clean_sentences = [clean_sentence(s) for s in sentences]
    return sentences, clean_sentences

# some functions from:
# https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
