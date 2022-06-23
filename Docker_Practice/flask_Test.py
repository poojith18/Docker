# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020

@author: krish.naik
"""

from flask import Flask, request
#import numpy as np
import pickle
import pandas as pd


app=Flask(__name__)
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Hello The answer is"+str(prediction)

#http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=2&entropy=1

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    #print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))

if __name__=='__main__':
    app.run()
    
    