import numpy as np
import re
import  csv

import self as self
from sklearn.metrics import precision_recall_fscore_support


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
def baca_csv_tanpa_regex(data_file):
    data = []
    with open(data_file) as f:
        r = csv.reader(f)
        for row in r:
            # print(row)
            data.append(row)
    return data

def baca_csv_set_hasil(data_file):
    data = []
    with open(data_file) as f:
        r = csv.reader(f)
        for row in r:
            # print(row)
            data.append(row[1])
    return data

class TestClass:
    def __init__(self):
        print ("in init")
    def testFunc(self):
        print ("in Test Func")

if __name__ == '__main__':
    # x = read_csv()
    # hasil_x = np.array(x)
    set_hasil = []
    set_aspek = []
    set_hasil.append(baca_csv_set_hasil('hasil.csv'))
    set_aspek.append(baca_csv_tanpa_regex('aspek.csv'))
    y_true = set_aspek
    y_pred = set_hasil
    print(y_true)
    print(y_pred)

    # precision_recall_fscore_support(y_true, y_pred, average='macro')
    variablex = len(y_true[0])
    salah = (300 - variablex)/300
    total = variablex/300
    # print(len(y_true[0]), len(y_pred[0]))
    print('Hasil terekstraksi yang sesuai: ')
    print(total)
    print('Hasil yang salah: ')
    print(salah)
    # precision_recall_fscore_support(y_true, y_pred, average='micro')
    # print(set_hasil)