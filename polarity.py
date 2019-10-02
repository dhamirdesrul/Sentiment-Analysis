import csv
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob.classifiers import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer

sentences=['light',
'durable-seeming',
'real',
'inconsistent',
'2mp',
'was',
'intelliflash',
'high-end',
'interesting',
'uncommon',
'inconvenient',
'compatible',
'same',
'effective',
'first',
'turned-on',
'narrow',
'outstanding',
'gorgeous',
'perfect',
'superior',
'wonderful',
'best',
'superb',
'great',
'wonderfully',
'excellent',
'happy',
'beautiful',
'pain',
'destroys',
'die',
'hate',
'horrific'
]
paragraph = [  
    'outstanding',
'gorgeous',
'perfect',
'superior',
'wonderful',
'best',
'superb',
'great',
'wonderfully',
'excellent',
'happy',
'beautiful',
'pain',
'destroys',
'die',
'hate',
'horrific',
'horrific',
'excellent',
'true',
'good',
'coolest',
'nice',
'dark',
'wrong',
'ok',
'dissapointing'
]
simpan = []
sentences.extend(paragraph)
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print('')
        if (ss[k] < 0):
            sentimen = 'negatif'
        elif (ss[k] == 0):
            sentimen = 'netral'
        elif (ss[k] > 0):
            sentimen = 'positif'
        fix = sentence, sentimen
        simpan.append(fix)
print(simpan)

x = []
with open('data_test.csv') as file:
    f = csv.reader(file)
    classifier = NaiveBayesClassifier(simpan)
    for i in f:
        a = classifier.classify(i)
        print(i, a)
        y = i, a
        x.append(y)


