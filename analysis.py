# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('IMDB_Dataset.csv')

# Preprocess the data
# Lowercase the reviews
df['review'] = df['review'].str.lower()

# Remove special characters
df['review'] = df['review'].str.replace('[^a-zA-Z]', ' ', regex=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training and evaluation functions
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
evaluate_model(nb, X_test_vec, y_test, "Naive Bayes")

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_vec, y_train)
evaluate_model(rf, X_test_vec, y_test, "Random Forest")

# Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train_vec, y_train)
evaluate_model(svm, X_test_vec, y_test, "SVM")

