#this takes a paragraph and tells you the probability of being bipolar vs.
#depressed. Also, based on whether you are a patient or a doctor
#it provides helpful links for next step
sentence_to_test = input('Input sentence to test for condition...')


import pandas as pd
import numpy as pd
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import webbrowser

#taking out any instances of page break
sentence_to_test = sentence_to_test.replace('\n', '')
#going to take out only letters, tokenize, lemmatize and remove
#stopwords and custom words


#these are manual words to remove from text to not make it easy
words_to_remove = ['depression', 'bipolar', 'antidepressants', 'manic', 'mania', 'hypomanic', 'hypomania']
#retain only letters
letters_only = re.sub("[^a-zA-Z]", " ", sentence_to_test)

tokenizer = RegexpTokenizer(r'\w+')#tokenize the data
text_tokens = tokenizer.tokenize(letters_only.lower())
#remove stopwords
remove_stopwords = [w for w in text_tokens if w not in stopwords.words('english')]

lemmatizer = WordNetLemmatizer()
text_lem = [lemmatizer.lemmatize(i) for i in remove_stopwords]

words_to_remove = [lemmatizer.lemmatize(i) for i in words_to_remove]
words_to_remove = set(words_to_remove)
#remove manual list of words that should be removed
remaining_words = [w for w in text_lem if w not in words_to_remove]

cleaned_text = [" ".join(remaining_words)]


pickle_in_cvec = open('cvec.pickle', 'rb')
cvec = pickle.load(pickle_in_cvec)
#Countvectorizing to pass in to the model
#cvec = CountVectorizer(max_features=5000, min_df=2, max_df=.8, ngram_range = (1,2))

transf_text = cvec.transform(cleaned_text)

#load in the pickled model
pickle_in = open('mnb.pickle', 'rb')
mnb = pickle.load(pickle_in)

predict = mnb.predict(transf_text)
predict_proba = mnb.predict_proba(transf_text)

print()
print('The odds of depression are {}%'.format((round(predict_proba[0,0], 1)*100)))
print()
print("The odds of bipolar are {}%".format((round(predict_proba[0,1], 1)*100)))


doc_pat = input('Are you a doctor or patient?')

if doc_pat == 'patient':
#give help links relative to whether they are depressed or bipolar
    if (round(predict_proba[0,1], 1)*100) > (round(predict_proba[0,0], 1)*100):
        webbrowser.open('https://www.amenclinics.com/conditions/bipolar-disorder/')
    else:
        webbrowser.open('https://www.amenclinics.com/conditions/anxiety-and-depression/')
if doc_pat == 'doctor':
    webbrowser.open('https://www.apa.org/education/ce/')
