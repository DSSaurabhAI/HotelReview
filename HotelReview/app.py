from flask import Flask
from flask import request
from flask import redirect
from flask import session
from flask import url_for
from flask import render_template

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
ps = PorterStemmer()
for index in range(len(dataset)):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][index])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, Y_train)



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
	if request.method=='POST':
		comment = request.form['comment']
		data = [comment]
		temp = cv.transform(data).toarray()
		prediction = classifier.predict(temp)
		return render_template('results.html', prediction=prediction)
	else:
		return render_template('index.html')

if __name__=='__main__':
	app.run(debug=True)