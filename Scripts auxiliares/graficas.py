import csv
import numpy as np
import matplotlib.pyplot as plt

base_chain = "../Ejecuciones_3_libre_albedrio/Ejecuciones_libre_albedrio/"
execution = 3

if execution == 0:
    cad1 = base_chain + 'Resultados_promediados/MedianHouseValue_Async_Aver_Fit.csv' 
    cad2 = base_chain + 'Resultados_promediados/MedianHouseValue_Async_Best_Fit.csv'
    cad3 = base_chain + 'Resultados_promediados/MedianHouseValue_Async_TOP_Fit.csv'
elif execution == 1:
    cad1 = base_chain + 'Resultados_promediados/MedianHouseValue_Sync_Aver_Fit.csv' 
    cad2 = base_chain + 'Resultados_promediados/MedianHouseValue_Sync_Best_Fit.csv' 
    cad3 = base_chain + 'Resultados_promediados/MedianHouseValue_Sync_TOP_Fit.csv' 
elif execution == 2:
    cad1 = base_chain + 'Resultados_promediados/FIFA_Async_Aver_Fit.csv' 
    cad2 = base_chain + 'Resultados_promediados/FIFA_Async_Best_Fit.csv' 
    cad3 = base_chain + 'Resultados_promediados/FIFA_Async_TOP_Fit.csv' 
elif execution == 3:
    cad1 = base_chain + 'Resultados_promediados/FIFA_Sync_Aver_Fit.csv' 
    cad2 = base_chain + 'Resultados_promediados/FIFA_Sync_Best_Fit.csv'
    cad3 = base_chain + 'Resultados_promediados/FIFA_Sync_TOP_Fit.csv'
    
a_1 = []
a_2 = []
a_3 = []

with open(cad1, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_1.append(float(', '.join(row)))

with open(cad2, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_2.append(float(', '.join(row)))

with open(cad3, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_3.append(float(', '.join(row)))

a_1 = np.array(a_1)
a_2 = np.array(a_2)
a_3 = np.array(a_3)

x = np.arange(0, len(a_3))

if execution == 0:
    plt.title("Sistema con ECO (California)")
elif execution == 1:
    plt.title("Sistema sin ECO (California)")
elif execution == 2:
    plt.title("Sistema con ECO (FIFA)")
elif execution == 3:
    plt.title("Sistema sin ECO (FIFA)")

plt.xlabel("Iteración del proceso evolutivo")

if execution == 0 or execution == 1:
    plt.ylim([0.2, 0.3])
    plt.ylabel("Mean Squared Error (MSE)")
else:
    plt.ylim([0.45, 0.85])
    plt.ylabel("Precisión (Accuracy)")

plt.plot(x, a_1, color ="red")
plt.plot(x, a_2, color ="blue")
plt.plot(x, a_3, color ="green")
plt.legend(['Grado de adaptación promedio', 'Grado de adaptación mejor individuo', 'Grado de adaptación promedio (TOP individuos)'])
plt.show()