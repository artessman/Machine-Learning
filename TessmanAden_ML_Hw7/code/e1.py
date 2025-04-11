
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

#only works for binomial
def NBbyhand(Class0Dic, Class1Dic, prior0, prior1, Classnames, test, alpha=1):
    
    Class0total = sum(Class0Dic.values())
    Class1total = sum(Class1Dic.values())
    
    Class0prob = []
    Class1prob = []
   
    for word in test: 
        Class0prob.append((Class0Dic.get(word, 0) + alpha) / (Class0total + alpha * len(Class0Dic)))
        Class1prob.append((Class1Dic.get(word, 0) + alpha) / (Class1total + alpha * len(Class1Dic)))
    
    #intialize prob with prior
    prob0 = prior0
    prob1 = prior1
    
    for value in Class0prob:
        prob0 *= value
    for value in Class1prob:
        prob1 *= value
    
    pred = 'NA'

    if prob0 > prob1:
       pred = Classnames[0]
    else:
       pred = Classnames[1]

    return (prob0, prob1, pred)


#Prior values
priorNorm = 0.73
priorSpam = 0.27

#Read files for plotting 
with open("train_N.txt", "r") as f:
    train_N = f.read().split()

with open("train_S.txt", "r") as f:
    train_S = f.read().split()

#count the frequency of words
countsN = Counter(train_N)
countsS = Counter(train_S)

# Extract the words and their frequencies
key_listN = list(countsN.keys())
val_listN = list(countsN.values())

key_listS = list(countsS.keys())
val_listS = list(countsS.values())

# read in test data
with open("testEmail_I.txt", "r") as f:
    test1 = f.read().split()

with open("testEmail_II.txt", "r") as f:
    test2 = f.read().split()

#my pred 
Classnames = ['Normal', 'Spam']

#test1
prob0, prob1, pred =  NBbyhand(countsN, countsS, priorNorm, priorSpam, Classnames, test1)
print(f'Test1 Prediction: {pred}')
print(f'Normal Probability: {prob0}')
print(f'Spam Probability: {prob1}')

# test2
prob0, prob1, pred =  NBbyhand(countsN, countsS, priorNorm, priorSpam, Classnames, test2)
print()
print(f'Test2 Prediction: {pred}')
print(f'Normal Probability: {prob0}')
print(f'Spam Probability: {prob1}')


#plots
figure, axis = plt.subplots(1, 2)

# Plot for Normal Words
axis[0].bar(key_listN, val_listN, color='blue')
axis[0].set_title("Normal Word Frequencies")
axis[0].set_xlabel("Words")
axis[0].set_ylabel("Frequency")
axis[0].tick_params(axis='x', rotation=45)# way to nake the words more legible

#Plot for Spam Words
axis[1].bar(key_listS, val_listS, color='red')
axis[1].set_title("Spam Word Frequencies")
axis[1].set_xlabel("Words")
axis[1].set_ylabel("Frequency")
axis[1].tick_params(axis='x', rotation=45) #way to nake the words more legible

# hopfully better spacing and readability
plt.tight_layout()

plt.show()