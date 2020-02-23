import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import CountVectorizer

#this data is already cleaned and pre_processed
df = pd.read_csv('../data/data_pre_processed.csv') #importing data to train on

df.dropna(inplace = True) #drop any null values

y = df['subreddit'].map({'depression': 0, 'bipolar':1})  #our y variable
X = df['title_selftext'] #our X variable

cvec = CountVectorizer(max_features = 500, min_df = 2, max_df = .8,
                    ngram_range = (1, 2))
X_cvec = cvec.fit_transform(X) #fit and transform with CountVectorizer

with open('cvec.pickle', 'wb') as f:
    pickle.dump(cvec, f) #make a pickle file to them be able to CountVectorizer
    #new data coming in

mnb = MultinomialNB() #fit our model and pickle it
mnb.fit(X_cvec, y)
with open('mnb.pickle', 'wb') as f:
    pickle.dump(mnb, f)

print('pickling done')
