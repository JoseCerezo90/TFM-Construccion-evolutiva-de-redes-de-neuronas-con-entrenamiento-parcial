import csv
import numpy as np
import matplotlib.pyplot as plt

execution = 0

if execution == 0: cad = './California/Entrenamiento_Individuo_Promediado.csv'
else: cad = './FIFA/Entrenamiento_Individuo_Promediado.csv'
    
a_1 = []

with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_1.append(float(', '.join(row)))

a_1 = np.array(a_1)
x = np.arange(0, 300)


plt.plot(x, a_1, color ="red")
plt.show()