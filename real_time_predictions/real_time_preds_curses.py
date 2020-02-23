#this takes real time input and provides odds of depression vs. Bipolar
#as you type in updating it as you go.
#acknowledgement to Will Sutton for help with the code.

import curses
import time
import sys
import numpy as np
import pandas as pd
import numpy as pd
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(2)

stdscr.addstr(0,10,"Real-time Model Scoring: press down-arrow to quit")
stdscr.addstr(20,50,"P(Depression): |       | ")
stdscr.addstr(21,50,"P(BipolarDiso): |       | ")
stdscr.addstr(3,1,'>')
stdscr.refresh()

stdscr.nodelay(True)
#load in the fitted coutvectorizer and the fitted model
pickle_in_cvec = open('cvec.pickle', 'rb')
cvec = pickle.load(pickle_in_cvec)

pickle_in = open('mnb.pickle', 'rb')
mnb = pickle.load(pickle_in)

#the function takes the input and cleanes and pre-process it
def prediction_model(sentence_to_test, cvec = cvec, mnb = mnb):
    sentence_to_test = sentence_to_test.replace('\n', '')
    words_to_remove = ['depression', 'bipolar', 'antidepressants', 'manic', 'mania', 'hypomanic', 'hypomania']

    letters_only = re.sub("[^a-zA-Z]", " ", sentence_to_test)
    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(letters_only.lower())
    remove_stopwords = [w for w in text_tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    text_lem = [lemmatizer.lemmatize(i) for i in remove_stopwords]

    words_to_remove = [lemmatizer.lemmatize(i) for i in words_to_remove]
    words_to_remove = set(words_to_remove)

    remaining_words = [w for w in text_lem if w not in words_to_remove]

    cleaned_text = [" ".join(remaining_words)]

    # pickle_in_cvec = open('cvec.pickle', 'rb')
    # cvec = pickle.load(pickle_in_cvec)

    transf_text = cvec.transform(cleaned_text)
    # pickle_in = open('mnb.pickle', 'rb')
    # mnb = pickle.load(pickle_in)
    predict = mnb.predict(transf_text)
    predict_proba = mnb.predict_proba(transf_text)

    p_d = (round(predict_proba[0,0], 1)*100)
    p_b = (round(predict_proba[0,1], 1)*100)
    return p_d, p_b

#the function updates to the screen the probability of being depressed vs
#bipolar
def update_screen(_stdscr, msg):

    p_d, p_b = prediction_model(msg)


    msg_a = f"P(Depresssion): | {p_d} | - |"
    msg_b = f"P(BipolarDiso): | {p_b} | - |"

    _stdscr.addstr(20,50,msg_a)
    _stdscr.addstr(21,50,msg_b)

key = ''
final = 'no final'
key_old = '>'
counter = 1
max_counter = 1
b_active = False
t0 = time.time()
msg = ''
n = 3

PAUSE_THRESH = 1
PAUSE_HI_THRESH = 10

while True:

    stdscr.nodelay(True)

    try:
        key = stdscr.getch()
    except:
        continue

    if (time.time() - t0) < PAUSE_THRESH:
        b_active = False

    elif PAUSE_HI_THRESH > (time.time() - t0) > PAUSE_THRESH:
        if not(b_active):
            update_screen(stdscr, msg)
            b_active = True
            stdscr.addch(n, counter, key_old)


    if key == curses.KEY_DOWN:
        break

    elif key != -1:
        counter += 1
        try:
            stdscr.addch(n,counter,key)
        except:
            n += 1
            counter = 0
            stdscr.addch(n, counter, key)
        msg += str(chr(key))
        key_old = key
        t0 = time.time()

curses.endwin()

print(msg)
