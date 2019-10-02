from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import csv
import json as j
import nltk

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def buka_file(filename):
    data = []
    with open(filename) as f:
        r = csv.reader(f)
        for row in r:
            data.append(row)
            print(row)
    return data


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
        sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
            vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(y):
    stop_words = stopwords.words('english')
    summarize_text = []
    dictionarynya = {
        'kelas 1' : [],
        'kelas 2' : [],
        'kelas 3' : [],
        'kelas 4' : [],
        'kelas 5' : [],
        'kelas 6' : [],
        'kelas 7' : [],
        'kelas 8' : [],
        'kelas 9' : [],
        'kelas 10': [],
        'kelas 11': [],
        'kelas 12': []
    }
    # Step 1 - Read text anc split it
    sentences =  y
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    print("Indexes of top ranked_sentence order are ", ranked_sentence)
    top_n = len(y)
    print(top_n)
    for i in range(top_n):
        if (i < 17):
            dictionarynya['kelas 1'].append(ranked_sentence[i][1])
        elif (i < 34):
            dictionarynya['kelas 2'].append(ranked_sentence[i][1])
        elif (i < 51):
            dictionarynya['kelas 3'].append(ranked_sentence[i][1])
        elif (i < 68):
            dictionarynya['kelas 4'].append(ranked_sentence[i][1])
        elif (i < 85):
            dictionarynya['kelas 5'].append(ranked_sentence[i][1])
        elif (i < 102):
            dictionarynya['kelas 6'].append(ranked_sentence[i][1])
        elif (i < 119):
            dictionarynya['kelas 7'].append(ranked_sentence[i][1])
        elif (i < 136):
            dictionarynya['kelas 8'].append(ranked_sentence[i][1])
        elif (i < 153):
            dictionarynya['kelas 9'].append(ranked_sentence[i][1])
        elif (i < 170):
            dictionarynya['kelas 10'].append(ranked_sentence[i][1])
        elif (i < 187):
            dictionarynya['kelas 11'].append(ranked_sentence[i][1])
        elif (i < 204):
            dictionarynya['kelas 12'].append(ranked_sentence[i][1])
    summarize_text.append((str(ranked_sentence[i][1])))
    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
    return summarize_text, dictionarynya

# let's begin
y = buka_file('seed.csv')
x = buka_file('hasil_regex.csv')
# print(y)
simpan, kelas = generate_summary(y)
# simpan1 = generate_summary(x)

print(simpan)
# print('')
print(kelas)

# print(simpan1)
# np.savetxt('summary_result.csv', simpan1, fmt='%s', delimiter=',')
np.savetxt('summary_result_seed.csv', simpan, fmt='%s', delimiter=',')
v = j.dumps(kelas)
f = open('hasil_kelas.json', "w")
f.write(v)
f.close()
# nltk.vader
