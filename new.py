import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, roc_curve, auc
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Data3.csv')

print(data.head())
print(data.info())

if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

data['tokenized_text'] = data['Body'].apply(lambda x: word_tokenize(x.lower()))
stop_words = set(nltk.corpus.stopwords.words('english'))
data['filtered_text'] = data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
stemmer = nltk.PorterStemmer()
data['stemmed_text'] = data['filtered_text'].apply(lambda x: [stemmer.stem(word) for word in x])
data['final_text'] = data['stemmed_text'].apply(lambda x: ' '.join(x))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['final_text'])
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
print(type(X_tfidf))
print("Shape of TF-IDF transformed data:", X_tfidf.shape)

X = X_tfidf
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

predictions = nb_classifier.predict(X_test)

num_folds = 5
accuracy_values = cross_val_score(nb_classifier, X, y, scoring='accuracy', cv=num_folds)
precision_values = cross_val_score(nb_classifier, X, y, scoring='precision_weighted', cv=num_folds)
recall_values = cross_val_score(nb_classifier, X, y, scoring='recall_weighted', cv=num_folds)
f1_values = cross_val_score(nb_classifier, X, y, scoring='f1_weighted', cv=num_folds)
conf_matrix = confusion_matrix(y_test, predictions)

print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")
print("Confusion Matrix:", conf_matrix)

long_string = ','.join(data['final_text'].tolist())
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('Word Cloud for Entire Dataset')
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=data['Label'].unique(), yticklabels=data['Label'].unique())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()

fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

actual_data = pd.read_csv('Data3.csv')

actual_data['tokenized_text'] = actual_data['Body'].apply(lambda x: word_tokenize(x.lower()))
actual_data['filtered_text'] = actual_data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
actual_data['stemmed_text'] = actual_data['filtered_text'].apply(lambda x: [stemmer.stem(word) for word in x])
actual_data['final_text'] = actual_data['stemmed_text'].apply(lambda x: ' '.join(x))

long_string_actual = ','.join(actual_data['final_text'].tolist())
wordcloud_actual = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
wordcloud_actual.generate(long_string_actual)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud_actual, interpolation="bilinear")
plt.axis("off")
plt.title('Word Cloud for Actual Data')
plt.show()

labels = ['Actual Spam', 'Actual Ham']
sizes_actual = [len(actual_data[actual_data['Label'] == 1]), len(actual_data[actual_data['Label'] == 0])]
sizes_predicted = [len(predictions[predictions == 1]), len(predictions[predictions == 0])]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].pie(sizes_actual, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
axs[0].set_title('Actual Data')
axs[1].pie(sizes_predicted, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
axs[1].set_title('Predicted Data')
plt.show()