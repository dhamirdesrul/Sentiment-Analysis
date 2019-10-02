import csv
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
sentences.extend(paragraph)
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
    if (ss['compound']<0.0):
        print(sentence ,': negatif')
    elif(ss['compound']>0.0):
        print(sentence ,': positif')
    else:
        print(sentence ,': netral')
        