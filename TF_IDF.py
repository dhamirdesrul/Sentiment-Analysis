import csv
from nltk.tokenize import RegexpTokenizer
import nltk
docA = "years wrong worth wonderfully wonderful"
docB = "you'll carry it with you every day you will definitly appreciate the small size you take around 12 pictures in a circle stitch them together and voila; you have a perfect memory of where you were standing you need to use the manual exposure adjustment in backlight or bright light conditions such as with snow in the background you have the option to have the flash automatic or off"

bowA = docA.split()
bowB = docB.split()

print(bowB)

wordSet = set(bowA).union(set(bowB))
print(wordSet)

wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

print(wordDictA)

for word in bowA:
    wordDictA[word] += 1

for word in bowB:
    wordDictB[word] += 1

print(wordDictA)

import pandas as pd
print(pd.DataFrame([wordDictA, wordDictB]))

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)

print(tfBowA)

print(tfBowB)


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

idfs = computeIDF([wordDictA, wordDictB])

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)

import pandas as pd
y = (pd.DataFrame([tfidfBowA, tfidfBowB]))
print(y)
print(tfidfBowB)
print(tfidfBowA)

sorting = (sorted(tfidfBowB.values(), reverse=True))
sorting1 = sorted(tfidfBowB.keys(), reverse=True)
# print(sorting1)
# print(sorting)
nilai = []
# for i in range(6):
#     nilai.append(sorting[i])
# print(nilai)
for key, value in sorted(tfidfBowB.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))
    nilai.append(("%s: %s" % (key, value)))
# print(nilai)

terbaik = []
for i in range(6):
    terbaik.append(nilai[i])
print(terbaik)