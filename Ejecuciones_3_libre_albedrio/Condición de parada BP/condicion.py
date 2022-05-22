import csv
import numpy as np
import matplotlib.pyplot as plt

execution = 0

if execution == 0: 
    cad = './California/Entrenamiento_Individuo_Promediado.csv'
    print("California")
else: 
    cad = './FIFA/Entrenamiento_Individuo_Promediado.csv'
    print("FIFA")
    
a_1 = []
with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_1.append(float(', '.join(row)))

# No mejora más de un 0.01% tras 5 épocas seguidas
no_mejora = 0
# Porcentaje de mejora obligado de una época a otra
porcentaje_mejora_california = 0.0175
porcentaje_mejora_fifa = 0.05

for i in range(1, len(a_1)):
    # California
    if execution == 0: 
        porcentaje_mejora = ((a_1[i-1]/a_1[i])-1)*100
        if porcentaje_mejora < porcentaje_mejora_california: no_mejora += 1
        else: no_mejora = 0
        if no_mejora == 5:
            print("Detener entrenamiento. Época BP: " + str(i))
    # FIFA
    else: 
        porcentaje_mejora = ((a_1[i]/a_1[i-1])-1)*100
        if porcentaje_mejora < porcentaje_mejora_fifa: no_mejora += 1
        else: no_mejora = 0
        if no_mejora == 5:
            print("Detener entrenamiento. Época BP: " + str(i))
    
# California: 0.0175% --> Detiene por primera vez en la época 148 (de 300 que hay en total)
# FIFA: 0.05% ----------> Detiene por primera vez en la época 124 (de 300 que hay en total)