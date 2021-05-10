import re
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet
import nltk

positive_reviews = open('imdb_pos.txt').read()
negative_reviews = open('imdb_neg.txt').read()
medium_positive_reviews = open('all_neg.txt').read()
medium_negative_reviews = open('all_pos.txt').read()
large_negative_reviews_test = open('large_neg_test.txt', encoding="utf8").read()
large_positive_reviews_test = open('large_pos_test.txt', encoding="utf8").read()
large_negative_reviews_train = open('large_neg_train.txt', encoding="utf8").read()
large_positive_reviews_train = open('large_pos_train.txt', encoding="utf8").read()

positive_sentences = [p for p in positive_reviews.split('\n') if p]
negative_sentences = [p for p in negative_reviews.split('\n') if p]

medium_negative_sentences = [p for p in medium_negative_reviews.split('\n') if p]
medium_positive_sentences = [p for p in medium_positive_reviews.split('\n') if p]

large_negative_sentences_test = [p for p in large_negative_reviews_test.split('\n') if p]
large_positive_sentences_test = [p for p in large_positive_reviews_test.split('\n') if p]
large_negative_sentences_train = [p for p in large_negative_reviews_train.split('\n') if p]
large_positive_sentences_train = [p for p in large_positive_reviews_train.split('\n') if p]

negative_review_text = []
positive_review_text = []
medium_negative_review_text = []
medium_positive_review_text = []

i = 0
for sentence in medium_negative_sentences:
    if '<review_text>' in sentence:
        medium_negative_review_text.append(medium_negative_sentences[i+1])
    i = i + 1

i = 0
for sentence in medium_positive_sentences:
    if '<review_text>' in sentence:
        medium_positive_review_text.append(medium_positive_sentences[i+1])
    i = i + 1

print("Data Loaded...")
positive_tokens = []
negative_tokens = []
medium_positive_tokens = []
medium_negative_tokens = []
large_negative_tokens_test = []
large_positive_tokens_test = []
large_negative_tokens_train = []
large_positive_tokens_train = []

print("Tokenizing Data...")
for sentence in positive_sentences:
    positive_tokens.append(word_tokenize(sentence))

for sentence in negative_sentences:
    negative_tokens.append(word_tokenize(sentence))

for sentence in medium_positive_sentences:
    medium_positive_tokens.append(word_tokenize(sentence))

for sentence in medium_negative_sentences:
    medium_negative_tokens.append(word_tokenize(sentence))

for sentence in large_negative_sentences_test:
    large_negative_tokens_test.append(word_tokenize(sentence))

for sentence in large_negative_sentences_train:
    large_negative_tokens_train.append(word_tokenize(sentence))

for sentence in large_positive_sentences_test:
    large_positive_tokens_test.append(word_tokenize(sentence))

for sentence in large_positive_sentences_train:
    large_positive_tokens_train.append(word_tokenize(sentence))


print("Cleaning data...")


def clean_data(my_data):
    processed_features = []
    for sentence in range(0, len(my_data)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(my_data[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        processed_feature = re.sub(r'[0-9]+', '', processed_feature)

        processed_feature = re.sub(r'[^\w\s]', '', processed_feature)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)

    return processed_features


med_neg_features = clean_data(medium_negative_review_text)
med_pos_features = clean_data(medium_positive_review_text)
large_pos_features_test = clean_data(large_positive_tokens_test)
large_neg_features_test = clean_data(large_negative_tokens_test)
large_pos_features_train = clean_data(large_positive_tokens_train)
large_neg_features_train = clean_data(large_negative_tokens_train)

english_stop_words = ENGLISH_STOP_WORDS

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in english_stop_words])
        )
    return removed_stop_words


med_neg_features = remove_stop_words(med_neg_features)
med_pos_features = remove_stop_words(med_pos_features)
large_pos_features_test = remove_stop_words(large_pos_features_test)
large_neg_features_test = remove_stop_words(large_neg_features_test)
large_pos_features_train = remove_stop_words(large_pos_features_train)
large_neg_features_train = remove_stop_words(large_neg_features_train)

med_neg_tokens = []
med_pos_tokens = []
large_pos_tokens_test = []
large_neg_tokens_test = []
large_pos_tokens_train = []
large_neg_tokens_train = []

for sentence in med_neg_features:
    med_neg_tokens.append(word_tokenize(sentence))

for sentence in med_pos_features:
    med_pos_tokens.append(word_tokenize(sentence))

for sentence in large_pos_features_test:
    large_pos_tokens_test.append(word_tokenize(sentence))

for sentence in large_neg_features_test:
    large_neg_tokens_test.append(word_tokenize(sentence))

for sentence in large_pos_features_train:
    large_pos_tokens_train.append(word_tokenize(sentence))

for sentence in large_neg_features_train:
    large_neg_tokens_train.append(word_tokenize(sentence))


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


lemmatizer = WordNetLemmatizer()


def lemmatize_data(my_data):
    single_review = []
    lemmed_data = []
    for review in my_data:
        for word in review:
            single_review.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        lemmed_data.append(' '.join(single_review))
        single_review = []
    return lemmed_data

print("Lemmatizing data...")
pos_lemmed_reviews = lemmatize_data(positive_tokens)
neg_lemmed_reviews = lemmatize_data(negative_tokens)
med_lemmed_neg = lemmatize_data(med_neg_tokens)
med_lemmed_pos = lemmatize_data(med_pos_tokens)


neg_tokens = []
pos_tokens = []
final_neg_med = []
final_pos_med = []
final_pos_test = []
final_neg_test = []
final_pos_train = []
final_neg_train = []


for sentence in pos_lemmed_reviews:
    pos_tokens.append(word_tokenize(sentence))

for sentence in neg_lemmed_reviews:
    neg_tokens.append(word_tokenize(sentence))

for sentence in med_lemmed_neg:
    final_neg_med.append(word_tokenize(sentence))

for sentence in med_lemmed_pos:
    final_pos_med.append(word_tokenize(sentence))


def get_reviews_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)


print("Generating tokens...")
positive_review_tokens_for_model = get_reviews_for_model(pos_tokens)
negative_review_tokens_for_model = get_reviews_for_model(negative_tokens)
med_neg_tokens_for_model = get_reviews_for_model(med_neg_tokens)
med_pos_tokens_for_model = get_reviews_for_model(final_pos_med)
large_neg_tokens_for_model_test = get_reviews_for_model(large_neg_tokens_test)
large_pos_tokens_for_model_test = get_reviews_for_model(large_pos_tokens_test)
large_neg_tokens_for_model_train = get_reviews_for_model(large_neg_tokens_train)
large_pos_tokens_for_model_train = get_reviews_for_model(large_pos_tokens_train)

positive_review_dataset = [(review_dict, "Positive")
                           for review_dict in positive_review_tokens_for_model]

negative_review_dataset = [(review_dict, "Negative")
                           for review_dict in negative_review_tokens_for_model]

negative_med_dataset = [(review_dict, "Negative")
                        for review_dict in med_neg_tokens_for_model]

positive_med_dataset = [(review_dict, "Positive")
                        for review_dict in med_pos_tokens_for_model]

negative_large_dataset_test = [(review_dict, "Negative")
                               for review_dict in large_neg_tokens_for_model_test]

positive_large_dataset_test = [(review_dict, "Positive")
                               for review_dict in large_pos_tokens_for_model_test]

negative_large_dataset_train = [(review_dict, "Negative")
                                for review_dict in large_neg_tokens_for_model_train]

positive_large_dataset_train = [(review_dict, "Positive")
                                for review_dict in large_pos_tokens_for_model_train]


print("Building datasets...")
imdb_dataset = positive_review_dataset + negative_review_dataset
random.shuffle(imdb_dataset)

med_dataset = positive_med_dataset + negative_med_dataset
random.shuffle(med_dataset)

large_dataset_test = negative_large_dataset_test + positive_large_dataset_test
random.shuffle(large_dataset_test)

large_dataset_train = negative_large_dataset_train + positive_large_dataset_train
random.shuffle(large_dataset_train)


print("Running model 1, Small Dataset...")
for i in range(0, 5):
    review_train_data, review_test_data = train_test_split(imdb_dataset, test_size=0.30, random_state=i)
    review_classifier = NaiveBayesClassifier.train(review_train_data)
    NB_accuracy = classify.accuracy(review_classifier, review_test_data)
    print("IMDB NB accuracy is:", NB_accuracy)

    review_classifier2 = SklearnClassifier(LogisticRegression())
    review_classifier2.train(review_train_data)

    acc_score = nltk.classify.accuracy(review_classifier2, review_test_data)
    print("IMDB LR accuracy is:", acc_score)


print("Running model 2, Medium Dataset...")
for i in range(0, 5):
    review_train_data, review_test_data = train_test_split(med_dataset, test_size=0.30, random_state=i)
    review_classifier = NaiveBayesClassifier.train(review_train_data)
    NB_accuracy = classify.accuracy(review_classifier, review_test_data)
    print("IMDB NB accuracy is:", NB_accuracy)

    review_classifier2 = SklearnClassifier(LogisticRegression())
    review_classifier2.train(review_train_data)

    acc_score = nltk.classify.accuracy(review_classifier2, review_test_data)
    print("IMDB LR accuracy is:", acc_score)


print("Running model 3, Large Dataset...")
for i in range(0, 5):
    review_classifier = NaiveBayesClassifier.train(large_dataset_train)
    NB_accuracy = classify.accuracy(review_classifier, large_dataset_test)
    print("Amazon reviews NB accuracy is:", NB_accuracy)

    review_classifier2 = SklearnClassifier(LogisticRegression(max_iter=1000))
    review_classifier2.train(large_dataset_train)
    acc_score = nltk.classify.accuracy(review_classifier2, large_dataset_test)
    print("Amazon reviews LR accuracy is:", acc_score)

