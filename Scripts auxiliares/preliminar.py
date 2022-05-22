import numpy as np
import csv

execution = 1
if execution == 1: 
    cadd1 = './Ejecuciones_preliminares/California/Asíncrono/MedianHouseValue_Async_Aver_Fit_'
    cadd2 = './Ejecuciones_preliminares/California/Asíncrono/MedianHouseValue_Async_Best_Fit_'
    cadd3 = './Ejecuciones_preliminares/California/Síncrono/MedianHouseValue_Sync_Aver_Fit_'
    cadd4 = './Ejecuciones_preliminares/California/Síncrono/MedianHouseValue_Sync_Best_Fit_'
else:
    cadd1 = './Ejecuciones_preliminares/FIFA/Asíncrono/FIFA_Async_Aver_Fit_'
    cadd2 = './Ejecuciones_preliminares/FIFA/Asíncrono/FIFA_Async_Best_Fit_'
    cadd3 = './Ejecuciones_preliminares/FIFA/Síncrono/FIFA_Sync_Aver_Fit_'
    cadd4 = './Ejecuciones_preliminares/FIFA/Síncrono/FIFA_Sync_Best_Fit_'

# Obtiene el último valor de cada csv
a_1 = []
a_aux = []
for i in range(10):
    cad = cadd1 + str(i+1) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_1.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

a_2 = []
a_aux = []
for i in range(10):
    cad = cadd2 + str(i+1) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_2.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

a_3 = []
a_aux = []
for i in range(10):
    cad = cadd3 + str(i+1) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_3.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

a_4 = []
a_aux = []
for i in range(10):
    cad = cadd4 + str(i+1) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_4.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []
    

print("## MSE/Precisión (Promedio) - SIN ECO ##")
print( str(round(np.mean(a_3),5)) + " & " + str(round(np.std(a_3),5)) + " & " + str(round(np.max(a_3),5)) + " & " + str(round(np.min(a_3),5)))
print("")
print("## MSE/Precisión (Mejor) - SIN ECO ##")
print( str(round(np.mean(a_4),5)) + " & " + str(round(np.std(a_4),5)) + " & " + str(round(np.max(a_4),5)) + " & " + str(round(np.min(a_4),5)))
print("")
print("## MSE/Precisión (Promedio) - CON ECO ##")
print( str(round(np.mean(a_1),5)) + " & " + str(round(np.std(a_1),5)) + " & " + str(round(np.max(a_1),5)) + " & " + str(round(np.min(a_1),5)))
print("")
print("## MSE/Precisión (Mejor) - CON ECO ##")
print( str(round(np.mean(a_2),5)) + " & " + str(round(np.std(a_2),5)) + " & " + str(round(np.max(a_2),5)) + " & " + str(round(np.min(a_2),5)))
