import csv
import sys
import matplotlib.pyplot as plt
import json
from matplotlib.pyplot import figure
import numpy as np
import math
from joblib import parallel_backend, Parallel, delayed
from sklearn.model_selection import train_test_split
from collections import Counter
import time
from sklearn import svm
import json

with open('data.json', 'r', encoding='utf-8') as f:  # открыли файл
    text = json.load(f)

bag_of_words = Counter(text).most_common(1000)


def has_cyrillic(text, ruRU=set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')):
    return not ruRU.isdisjoint(text.lower())


f = open("train.csv", encoding='utf-8')
rows = [line.split("\t") for line in f]
f.close()
print("train loaded")

X = []
y = []
start_time = time.time()

#@Oltyz

def load_fun(i):
    global rows
    wording = []
    row = rows[i]
    word = row[2].split(" ")
    if row[2] == '':
        return
    for j in range(0, len(word) - 1, 2):
        if has_cyrillic(word[j]):
            if not (word[j] in wording):
                wording.append(word[j])
    vector = []
    for word in bag_of_words:
        vector.append(int(word[0] in wording))
    global X, y
    X.append(vector)
    y.append(int(row[3]))
    if i%10000==0:
        print(i)
        print(time.time() - start_time)


Parallel()(delayed(load_fun)(i) for i in range(1, len(rows)))
# for i in range(1, len(rows)):
#     load_fun(i)
print("END", time.time() - start_time)

with parallel_backend('threading', n_jobs=6):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = svm.SVC()

    clf.fit(X_train, y_train)
    print(time.time() - start_time)
    predictions = clf.predict(X_test)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and y_test[i] == 1:
            tp += 1
        elif predictions[i] == 0 and y_test[i] == 1:
            fn += 1
        elif predictions[i] == 1 and y_test[i] == 0:
            fp += 1
        else:
            tn += 1
    print("TPR = ", tp / (tp + fn))
    print("FPR = ", fp / (fp + tn))
