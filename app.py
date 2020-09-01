# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:31:48 2020

@author: This PC
"""

# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'restaurant-sentiment-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

app=Flask(__name__)


@app.route('/')
def home():
	return render_template('index1.html')
    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	pred = classifier.predict(vect)[0]
    	return render_template('result.html', prediction=pred)
    
if __name__ == '__main__':
	app.run(debug=True)    