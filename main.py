import re
import string
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


positive_reviews = open('imdb_pos.txt').read()
negative_reviews = open('imdb_neg.txt').read()
positive_tokens = []
negative_tokens = []

positive_sentences = [p for p in positive_reviews.split('\n') if p]
negative_sentences = [p for p in negative_reviews.split('\n') if p]


for sentence in positive_sentences:
    positive_tokens.append(word_tokenize(sentence))

for sentence in negative_sentences:
    negative_tokens.append(word_tokenize(sentence))

stop_words = stopwords.words('english')

# Clean data


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# Create test/training sets

def get_reviews_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)


positive_review_tokens_for_model = get_reviews_for_model(positive_tokens)
negative_review_tokens_for_model = get_reviews_for_model(negative_tokens)

positive_review_dataset = [(review_dict, "Positive")
                           for review_dict in positive_review_tokens_for_model]

negative_review_dataset = [(review_dict, "Negative")
                           for review_dict in negative_review_tokens_for_model]

review_dataset = positive_review_dataset + negative_review_dataset

random.shuffle(review_dataset)


review_train_data = review_dataset[:700]
review_test_data = review_dataset[700:]

review_classifier = NaiveBayesClassifier.train(review_train_data)

print("IMDB accuracy is:", classify.accuracy(review_classifier, review_test_data))

