import matplotlib.pyplot as plt
import numpy as np
import csv

base_chain = "../Ejecuciones_3_libre_albedrio/Ejecuciones_libre_albedrio/"
execution = 3

if execution == 0:
    cadd1 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Epochs_'
    cadd2 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Epochs_'
elif execution == 1:
    cadd1 = base_chain + 'FIFA/Asincrono/FIFA_Async_Epochs_'
    cadd2 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Epochs_'
elif execution == 2:
    cadd1 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Best_Fit_'
    cadd2 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Best_Fit_'
elif execution == 3:
    cadd1 = base_chain + 'FIFA/Asincrono/FIFA_Async_Best_Fit_'
    cadd2 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Best_Fit_'

# Obtiene el último valor de cada csv
array1 = []
a_aux = []
for i in range(25):
    cad = cadd1 + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        array1.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

array2 = []
a_aux = []
for i in range(25):
    cad = cadd2 + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        array2.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

my_dict = {'Sistema sin ECO (secuencial)': array2, 'Sistema con ECO (concurrente)': array1}

fig, ax = plt.subplots()
if execution == 0: plt.title("Comparación épocas de entrenamiento (California)")
elif execution == 1: plt.title("Comparación épocas de entrenamiento (FIFA)")
elif execution == 2: plt.title("Comparación grado adaptación mejor individuo (California)")
elif execution == 3: plt.title("Comparación grado adaptación mejor individuo (FIFA)")

if execution == 0 or execution == 1: plt.ylabel("Número de épocas de entrenamiento")
elif execution == 2: plt.ylabel("Mean Squared Error (MSE)")
elif execution == 3: plt.ylabel("Precisión (Accuracy)")
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())

plt.show()