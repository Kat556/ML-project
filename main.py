import pandas as pd
import sys
import re
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.stdout = open('knnexample.txt', 'w')

imdb_reviews = pd.read_csv('imdb_labelled.txt', sep='\t', header=None)
imdb_reviews.columns = ['comments', 'ranking']

features = imdb_reviews.comments.values
labels = imdb_reviews.ranking.values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove all single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

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

vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
lemmed_reviews = vectorizer.fit_transform(lemmed_reviews).toarray()

X_train, X_test, y_train, y_test = train_test_split(lemmed_reviews, labels, test_size=.2, random_state=219)

scaler = StandardScaler()
scaler.fit(X_train, y_train)

accuracies = []

for i in range(2,151):
    print('\nk =',i)

    classifier = KNeighborsClassifier(n_neighbors=i, metric = 'jaccard')
    classifier.fit(X_train, y_train)
    
    y_prediction = classifier.predict(X_test)
    
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))
    print(accuracy_score(y_test, y_prediction))
    accuracies.append(accuracy_score(y_test, y_prediction))
    
curve = pd.DataFrame(accuracies, columns = ['accuracy'])
curve.index.name = i
a = curve.plot(title = 'graph')
a.set_ylabel('accuracy_score')
a.set_xlabel('k_values')


#classifier = KNeighborsClassifier(n_neighbors=45, metric='minkowski', algorithm='kd_tree', leaf_size=1)

#classifier.fit(X_train, y_train)

#y_prediction = classifier.predict(X_test)

#print(confusion_matrix(y_test, y_prediction))
#print(classification_report(y_test, y_prediction))
#print(accuracy_score(y_test, y_prediction))

