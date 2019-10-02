import csv
import re
import numpy as np

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

if __name__ == '__main__':
    x = read_csv()
    np.savetxt('hasil_regex.csv', x, fmt='%s', delimiter=',')