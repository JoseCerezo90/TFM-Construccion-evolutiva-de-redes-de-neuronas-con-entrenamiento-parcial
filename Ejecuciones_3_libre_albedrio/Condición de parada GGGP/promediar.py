import csv
import pandas as pd
import numpy as np

execution = 1

if execution == 0: cadd = './California/Mejores_Individuos_'
else: cadd = './FIFA/Mejores_Individuos_'


array = np.zeros((10, 21))

for i in range(10):
    cad = cadd + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        j = 0
        for row in spamreader:
            array[i][j] = ', '.join(row)
            j += 1

new_array = []
for i in range(21):
    accum = 0
    for j in range(10):
        accum += array[j][i]
    accum /= 10
    new_array.append(accum)

df = pd.DataFrame(new_array)

if execution == 0: df.to_csv('./California/Mejores_Individuos_Promediado.csv', index=False, header=False)
else: df.to_csv('./FIFA/Mejores_Individuos_Promediado.csv', index=False, header=False)