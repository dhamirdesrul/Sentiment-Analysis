import csv
import numpy as np
from gensim.summarization import summarize
import nltk
from nltk.sentiment import SentimentAnalyzer
from self import self
from textblob.classifiers import NaiveBayesClassifier

train_data = [  
    ('outstanding','positif'),
('gorgeous','positif'),
('perfect','positif'),
('superior','positif'),
('wonderful','positif'),
('best','positif'),
('superb','positif'),
('great','positif'),
('wonderfully','positif'),
('excellent','positif'),
('happy','positif'),
('beautiful','positif'),
('pain','negatif'),
('destroys','negatif'),
('die','negatif'),
('hate','negatif'),
('horrific','negatif'),
('horrific','negatif'),
('excellent','positif'),
('true','positif'),
('good','positif'),
('coolest','positif'),
('nice','positif'),
('dark','negatif'),
('wrong','negatif'),
('ok','positif'),
('dissapointing','negatif')
]

x = []
with open('data_test.csv') as file:
	f=csv.reader(file)
	classifier = NaiveBayesClassifier(train_data)
	for i in f:
		a=classifier.classify(i)
		print(i,a)
		y = i,a
		x.append(y)
		# print (summarize(i))

# sentim_analyzer = SentimentAnalyzer()
# trainer = NaiveBayesClassifier.train
# classifier = sentim_analyzer.train(trainer, train_data)
# # Training classifier
# for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
# 	print('{0}: {1}'.format(key, value))

print(x)
print(train_data)
print(len(train_data))
print(len(x))
sentim_analyzer = SentimentAnalyzer()
train_set = sentim_analyzer.apply_features(train_data)
test_set = sentim_analyzer.apply_features(x)
trainer = NaiveBayesClassifier.train
# classifier = sentim_analyzer.train(trainer, train_set)
