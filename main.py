import pandas as pd
import re
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression

imdb_reviews = pd.read_csv('imdb_labelled.txt', sep='\t', header=None)
imdb_reviews.columns = ['comments', 'ranking']

features = imdb_reviews.comments.values
labels = imdb_reviews.ranking.values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

english_stop_words = ENGLISH_STOP_WORDS

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in english_stop_words])
        )
    return removed_stop_words

processed_features = remove_stop_words(processed_features)

single_review = []
stemmed_reviews = []
stemmer = PorterStemmer()
for review in processed_features:
    for word in review.split():
        single_review.append(stemmer.stem(word))
    stemmed_reviews.append(' '.join(single_review))
    single_review = []

single_review = []
lemmed_reviews = []
lemmatizer = WordNetLemmatizer()
for review in stemmed_reviews:
    for word in review.split():
        single_review.append(lemmatizer.lemmatize(word))
    lemmed_reviews.append(' '.join(single_review))
    single_review = []

ngram_vector = CountVectorizer(binary=True, ngram_range=(1, 2))
lemmed_reviews = ngram_vector.fit_transform(lemmed_reviews).toarray()

X_train, X_test, y_train, y_test = train_test_split(lemmed_reviews, labels, train_size=.75, random_state=97)

lr = LogisticRegression(C=4)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

