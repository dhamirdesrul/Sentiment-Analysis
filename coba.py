'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/
'''
from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import numpy as np
import nltk
import csv
import re
import pandas as pd

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens


def read_csv():
    data = []
    with open('Dataset Canon S100 Reanotasi.csv') as f:
        r = csv.reader(f)
        for row in r:
            # print(row[0].split())

            a = re.sub(r'[#,]', ' ', row[0])
            a = a.split()
            i = 0
            while i < len(a):
                if '[' in a[i]:
                    a.remove(a[i])
                else:
                    i += 1
            # print(' '.join(a))
            data.append(' '.join(a))
    return data

def doublepropagasi():
    return "oke"

def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    x = values[m], counts[m]
    return x

def Rule11(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array):
    kata1 = ''
    kata2 = ''
    for i in dep_parser:
        if (i[0] == 'amod'):
            simpan_array.append(i)
            x = np.int(i[1]) - np.int(i[2])
            y = np.int(i[2])
            # print(tokenisasi[y-1])
            # print(y)
            if ((x) > 0):
                # print(i)
                tag_fix = tagger[y]
                # print(tag_fix)
                tot = text, tokenisasi[y]
                if (tag_fix[1] == 'NN'):
                    aspek_array.append(tokenisasi[y])
                    kata1 = tokenisasi[y]
                # print(tag_fix[1])
                if ((tokenisasi[y-1] not in seed) and (tokenisasi[y-1] not in seed)):
                    if (tag_fix[1]== 'JJ'):
                        seed.append(tokenisasi[y-1])
                        # print(tokenisasi[y - 1])
                        j = tokenisasi[y-1]
                        kata2 = tokenisasi[y-1]
                    else:
                        for j in tagger:
                            if (j[1] == 'JJ'):
                                if (j[0] not in seed):
                                    seed.append(j[0])
                                    kata2 = j[0]
    hasil = kata1, kata2
    hasil_array.append(hasil)

def Rule12(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array):
    kata1 = ''
    kata2 = ''
    for i in dep_parser:
        if (i[0] == 'nsubj' or i[0] == 'amod'):
            simpan_array.append(i)
            x = np.int(i[1]) - np.int(i[2])
            y = np.int(i[2])
            # print(tokenisasi[y-1])
            # print(y)
            tag_fix = tagger[y-1]
            u = tagger[y]
            if ((x) > 0 and (i[0] == 'nsubj')):
                # print(i)
                tot = text, tokenisasi[y - 1]
                if (tag_fix[1] == 'NN'):
                    # print(tag_fix[1])
                    aspek_array.append(tokenisasi[y - 1])
                    kata1 = tokenisasi[y-1]
                # else:
                #     for j in tagger:
                #         if (j[1] == 'NN'):
                #             if (j[0] not in aspek_array):
                #                 aspek_array.append(j[0])
            elif ((x) > 0 or (i[0] == 'amod')):
                # print(tokenisasi[y - 1])
                # if ((tokenisasi[y - 1] not in seed)):
                if (tag_fix[1] == 'JJ' or tag_fix[1] == 'JJS'):
                    seed.append(tokenisasi[y - 1])
                    kata2 = tokenisasi[y-1]
                    # print(kata2)
                else:
                    for j in tagger:
                        if (j[1] == 'JJ'):
                            seed.append(j[0])
                            kata2 = j[0]

    hasil = kata1, kata2
    hasil_array.append(hasil)

def Rule21(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array):
    kata1 = ''
    kata2 = ''
    for i in dep_parser:
        for j in hasil_array:
            if (j[0] not in aspek_array):
                # print(j[0])
                aspek_array.append(j[0])
                kata1 = j[0]
            else:
                if (j[1] not in seed):
                    seed.append(j[1])
                    kata2 = j[1]
    hasil = kata1, kata2
    hasil_array.append(hasil)

def Rule22(simpan_array, aspek_array, seed, dep_parser, tokenisasi, hasil_array):
    kata1 = ''
    kata2 = ''

    for i in dep_parser:
        for j in hasil_array:
            if (j[1] not in seed):
                seed.append(j[1])
                kata2 = j[1]
            else:
                if(j[0] not in aspek_array):
                    aspek_array.append(j[0])
                    kata1 = j[0]
                # print(j[1])
                # kata1 = j[0]
                # kata2 = j[1]
    hasil = kata1, kata2
    hasil_array.append(hasil)

def Rule31(simpan_array, aspek_array, tokenisasi, dep_parser, hasil_array):
    kata1 = ''
    kata2 = ''
    for i in dep_parser:
        if (i[0] == 'conj'):
            simpan_array.append(i)
            # print(i[0]
            x = np.int(i[1]) - np.int(i[2])
            y = np.int(i[1])
            z = np.int(i[2])
            tag_fix = tagger[y-1]

            # print(x)
            if (x < 0):
                # print(tokenisasi[y-1])
                # print(tokenisasi[z-1])
                p = text, tokenisasi[y-1]
                q = text, tokenisasi[z-1]
                if (tokenisasi[y-1] not in aspek_array):
                    if (tag_fix[1] == 'NN'):
                        # print(tokenisasi[y-1])
                        # print(tokenisasi[z-1])
                        # aspek_array.append(tokenisasi[y-1])
                        aspek_array.append(tokenisasi[z-1])
                        kata1 = tokenisasi[z-1]
                    else:
                        for j in tagger:
                            if (j[1] == 'NN'):
                                if (j[0] not in aspek_array):
                                    aspek_array.append(j[0])
                                    print(j[0])
                                    kata1 = j[0]
    hasil = kata1, kata2
    hasil_array.append(hasil)
def Rule32(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array):
    kata1 = ''
    kata2 = ''
    # arah1 = 0, 0
    # arah2 = 0, 0
    # for i in dep_parser:
    #     if (i[0] == 'nsubj'):
    #         # nsubj = 'nsubj'
    #         arah1 = i[1], i[2]
    #     if (i[0] == 'dobj'):
    #         arah2 = i[1], i[2]
    #         # dobj = 'dobj'
    # if (arah1[0] == arah2[0]):
    #     x = np.int(arah1[0]) - 2
    #     print(x)
    #     print(tokenisasi[x])
    # elif (arah1[0] != arah2[0]):
    #     x = np.int(arah2[0])
    #     print(x)
    #     print(tokenisasi[x])
    # else:
    #     print(tokenisasi[x])
    # elif (arah1[0] == 0):
    y = ''
    a = ''
    z = ''
    b = ''
    for i in dep_parser:
        if (i[0] == 'nsubj' or i[0] == 'dobj'):
            simpan_array.append(i)
            if (i[0] == 'nsubj'):
                y = np.int(i[1])
                a = np.int(i[2])
                # print(y)
            elif(i[0] == 'dobj'):
                z = np.int(i[1])
                b = np.int(i[2])
                # print(z)
            x = np.int(i[1]) - np.int(i[2])
            # tag_fix = tagger[a - 1]
            if (x < 0 and y == z):
                if (a > b):
                    # print(a-1)
                    # print(tokenisasi[y])
                    # print(tokenisasi[a-1])
                    tag_fix = tagger[a-1]
                    if (tokenisasi[a - 1] not in aspek_array):
                        if (tag_fix[1] == 'NN'):
                            # tot = text, tokenisasi[a - 1]
                            aspek_array.append(tokenisasi[a - 1])
                            kata1 = tokenisasi[a-1]
                        else:
                            for j in tagger:
                                if (j[1] == 'NN'):
                                    if (j[0] not in aspek_array):
                                        aspek_array.append(j[0])
                                        kata1 = tokenisasi[a-1]
                elif (b > a):
                    # print(b-1)
                    tag_fix = tagger[b - 1]
                    if (tokenisasi[b - 1] not in aspek_array):
                        if (tag_fix[1] == 'NN'):
                            aspek_array.append(tokenisasi[b-1])
                            kata1 = tokenisasi[b-1]
                        # else:
                        #     for j in tagger:
                        #         if (j[1] == 'NN'):
                        #             if (j[0] not in aspek_array):
                        #                 aspek_array.append(j[0])
        else:
            if ((y == '') or (z == '')):
                # print('tidak masuk')
                hasil = kata1, kata2
                hasil_array.append(hasil)
    hasil = kata1, kata2
    hasil_array.append(hasil)

def Rule41(simpan_array, aspek_array, seed, tokenisasi, dep_parser, tagger, hasil_array):
    kata1 = ''
    kata2 = ''
    for i in dep_parser:
        if (i[0] == 'conj'):
            simpan_array.append(i)
            # print(i[0])
            x = np.int(i[1]) - np.int(i[2])
            y = np.int(i[1])
            z = np.int(i[2])
            # print(x)
            if (x < 0):
                # print(tokenisasi[y-1])
                tag_fix = tagger[y-1]
                if(tokenisasi[y-1] not in seed):
                    if (tag_fix[1] == 'JJ'):
                        seed.append(tokenisasi[y-1])
                        kata2 = tokenisasi[y-1]
                    else:
                        for j in tagger:
                            if (j[1] == 'JJ'):
                                if (j[0] not in seed):
                                    seed.append(j[0])
                                    kata2 = tokenisasi[y-1]
                # aspek_array.append(tokenisasi[z-1])
        hasil = kata1, kata2
        hasil_array.append(hasil)
def Rule42(simpan_array, aspek_array, seed, tokenisasi, tagger, dep_parser, hasil_array):
    kata1 = ''
    kata2 = ''
    for i in dep_parser:
        if (i[0] == 'amod'):
            simpan_array.append(i)
            a = np.int(i[2])
            tag_fix = tagger[a-1]
            if (tag_fix[1] == "JJ" or tag_fix[1] == 'JJS'):
                if (tokenisasi[a-1] not in seed):
                    seed.append(tokenisasi[a-1])
                    kata2 = tokenisasi[a-1]
                    print(kata2)
    hasil = kata1, kata2
    hasil_array.append(hasil)

def buka_file():
    data = []
    with open('summary_result.csv') as f:
        r = csv.reader(f)
        for row in r:
            data.append(row)
            # print(row)
    return data

if __name__ == '__main__':
    sNLP = StanfordNLP()
    # text = "If you want to buy a sexy cool accessory-available mp3 player, you can choose iPod"
    # print("POS:", sNLP.pos(text))
    # print ("Tokens:", sNLP.word_tokenize(text))
    # print ("Dep Parse:", sNLP.dependency_parse(text))
    # tagger = sNLP.pos(text)
    # tokenisasi = nltk.word_tokenize(text)
    # print(tokenisasi)
    # dep_parser = np.array(sNLP.dependency_parse(text))
    # print(dep_parser)
    simpan_array = []
    aspek_array = []
    hasil_array = []
    seed = ['perfect', 'outstanding', 'great', 'wonderfully','best','hate', 'die', 'horrific', 'pain', 'destroys']
    data = []
    simpan_array = []
    aspek_array = []
    hasil_array = []
    seed = ['perfect', 'outstanding', 'great', 'wonderfully', 'best', 'hate', 'die', 'horrific', 'pain', 'destroys']
    data = read_csv()
    for i in data:
        text = i
        # print("POS:", sNLP.pos(text))
        # print("Tokens:", sNLP.word_tokenize(text))
        # print("Dep Parse:", sNLP.dependency_parse(text))
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule11(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule12(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule21(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule22(simpan_array, aspek_array, seed, dep_parser, tokenisasi, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule31(simpan_array, aspek_array, tokenisasi, dep_parser, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule32(simpan_array, aspek_array, seed, tokenisasi, dep_parser, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule41(simpan_array, aspek_array, seed, tokenisasi, dep_parser, tagger, hasil_array)

    for i in data:
        text = i
        tagger = sNLP.pos(text)
        tokenisasi = nltk.word_tokenize(text)
        dep_parser = np.array(sNLP.dependency_parse(text))
        Rule42(simpan_array, aspek_array, seed, tokenisasi, tagger, dep_parser, hasil_array)


    a = list(set(aspek_array))
    b = list(set(seed))
    c  = list(set(hasil_array))

    # np.savetxt('aspek.csv', a, delimiter=',', fmt='%s')
    np.savetxt('not_set_aspek.csv', aspek_array, delimiter=',', fmt='%s')
    # np.savetxt('seed.csv', b, delimiter=',', fmt='%s')
    np.savetxt('hasil.csv', hasil_array, delimiter=',', fmt='%s')
    # np.savetxt('sethasil.csv', c, delimiter=',', fmt='%s')
    # np.savetxt('parser.csv', dep_parser, delimiter=',', fmt='%s')
    # print(seed)
    print('')
    # print(a)
    # print(b)
    # print(c)
    print(aspek_array)
    print('')
    print(hasil_array)

    print(buka_file())






