import csv
import numpy as np
import matplotlib.pyplot as plt

execution = 1

if execution == 0: cad = './California/Mejores_Individuos_Promediado.csv'
else: cad = './FIFA/Mejores_Individuos_Promediado.csv'
    
a_1 = []

with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_1.append(float(', '.join(row)))

a_1 = np.array(a_1)
x = np.arange(0, 21)


plt.plot(x, a_1, color ="red")
plt.show()