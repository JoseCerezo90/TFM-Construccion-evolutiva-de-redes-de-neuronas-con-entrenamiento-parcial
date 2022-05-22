import csv
import pandas as pd
import numpy as np

execution = 0

if execution == 0: cadd = './California/Entrenamiento_Individuo_'
else: cadd = './FIFA/Entrenamiento_Individuo_'


array = np.zeros((50, 300))

for i in range(50):
    cad = cadd + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        j = 0
        for row in spamreader:
            array[i][j] = ', '.join(row)
            j += 1

new_array = []
for i in range(300):
    accum = 0
    for j in range(50):
        accum += array[j][i]
    accum /= 50
    new_array.append(accum)

df = pd.DataFrame(new_array)

if execution == 0: df.to_csv('./California/Entrenamiento_Individuo_Promediado.csv', index=False, header=False)
else: df.to_csv('./FIFA/Entrenamiento_Individuo_Promediado.csv', index=False, header=False)