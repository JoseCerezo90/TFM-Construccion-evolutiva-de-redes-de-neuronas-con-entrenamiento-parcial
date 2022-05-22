import csv
import numpy as np
import matplotlib.pyplot as plt

execution = 0

if execution == 0: 
    cad = './MedianHouseValue_Sync_Aver_Fit.csv'
    print("California")
else: 
    cad = './FIFA_Sync_Aver_Fit.csv'
    print("FIFA")
    
a_1 = []
with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_1.append(float(', '.join(row)))

porcentajes_mejora = []
for i in range(1, len(a_1)):
    # California
    if execution == 0: 
        porcentaje_mejora = ((a_1[i-1]/a_1[i])-1)*100
    # FIFA
    else: 
        porcentaje_mejora = ((a_1[i]/a_1[i-1])-1)*100
    
    if porcentaje_mejora > 0: porcentajes_mejora.append(porcentaje_mejora)

# Me quedo los últimos 10 porcentajes de mejora (porque la población ya avanza de una forma mucho más lenta porque ya han pasado muchas iteraciones GGGP)
new_porcentajes_mejora = porcentajes_mejora[len(porcentajes_mejora)-10:]
print(new_porcentajes_mejora)
print("Porcentaje de mejora promedio: " + str(np.mean(new_porcentajes_mejora)))

# California --> Porcentaje de mejora promedio: 0.12853812419951538 --> 0.129
# FIFA --> Porcentaje de mejora promedio: 0.11491954091946477 --> 0.115

# En la condición no se va a usar toda la población, como en el caso de arriba, sino a los top-mejores, por lo tanto, el porcentaje de mejora, de forma general,
# va a ser más elevado. Esto hace que sea más difícil que cumpla la condición de parada, por eso los porcentajes van a ser los de arriba (si los porcentajes fueran más
# bajos, es más difícil que se cumpla la condición), tras 5 iteraciones seguidas.