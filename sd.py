import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(words_):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    words_ = re.sub(r'\$\w*', '', words_)
    words_ = re.sub(r'^RT[\s]+', '', words_)
    words_ = re.sub(r'https?:\/\/.*[\r\n]*', '', words_)   
    words_ = re.sub(r'#', '', words_)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    words_tokens = tokenizer.tokenize(words_)

    words_clean = []
    for word in words_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            words_clean.append(stem_word)

    return words_clean


def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
