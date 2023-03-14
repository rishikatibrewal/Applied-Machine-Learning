import numpy as np
import pandas as pd
import sklearn
import joblib
from sentence_transformers import SentenceTransformer
sen_model = SentenceTransformer('all-MiniLM-L6-v2')

filename = open("C:/Users/Rishika Tibrewal/OneDrive/Desktop/Applied-Machine-Learning/Assignment 2/mlpmodel.joblib",'rb')
mlp =joblib.load(filename)

def score(text:str, model, threshold:float=0.5) -> (bool,float):
    # Transform the input text using the same used during training
    emb = sen_model.encode([text])
    # Predict the propensity score for the input text for each class
    prediction=model.predict(emb)
    propensity = model.predict_proba(emb)[:,1]
    return prediction[0], propensity[0]