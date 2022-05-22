import numpy as np
import csv

# REF: https://www.geeksforgeeks.org/python-convert-list-characters-string/
def convert(s):
    # initialization of string to ""
    new = ""
    # traverse in the string 
    for x in s:
        new += x 
    # return string 
    return new

base_chain = "../Ejecuciones_3_libre_albedrio/Ejecuciones_libre_albedrio/"
execution = 1

if execution == 1: 
    cadd = base_chain + 'California/Asincrono/MedianHouseValue_Async_'
    cadd1 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Aver_Fit_'
    cadd2 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Best_Fit_'
    cadd3 = base_chain + 'California/Asincrono/MedianHouseValue_Async_TOP_Fit_'
    print("###############################################")
    print("# RESULTADOS CALIFORNIA CONCURRENTE (CON ECO) #")
    print("###############################################")
    print()

elif execution == 2:
    cadd = base_chain + 'California/Sincrono/MedianHouseValue_Sync_'
    cadd1 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Aver_Fit_'
    cadd2 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Best_Fit_'
    cadd3 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_TOP_Fit_'
    print("##############################################")
    print("# RESULTADOS CALIFORNIA SECUENCIAL (SIN ECO) #")
    print("##############################################")
    print()

elif execution == 3: 
    cadd = base_chain + 'FIFA/Asincrono/FIFA_Async_'
    cadd1 = base_chain + 'FIFA/Asincrono/FIFA_Async_Aver_Fit_'
    cadd2 = base_chain + 'FIFA/Asincrono/FIFA_Async_Best_Fit_'
    cadd3 = base_chain + 'FIFA/Asincrono/FIFA_Async_TOP_Fit_'
    print("#########################################")
    print("# RESULTADOS FIFA CONCURRENTE (CON ECO) #")
    print("#########################################")
    print()

elif execution == 4: 
    cadd = base_chain + 'FIFA/Sincrono/FIFA_Sync_'
    cadd1 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Aver_Fit_'
    cadd2 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Best_Fit_'
    cadd3 = base_chain + 'FIFA/Sincrono/FIFA_Sync_TOP_Fit_'
    print("########################################")
    print("# RESULTADOS FIFA SECUENCIAL (SIN ECO) #")
    print("########################################")
    print()

# Obtiene el último valor de cada csv
a_1 = []
a_aux = []
for i in range(25):
    cad = cadd1 + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_1.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

a_2 = []
a_aux = []
for i in range(25):
    cad = cadd2 + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_2.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

a_3 = []
a_aux = []
for i in range(25):
    cad = cadd3 + str(i) + '.csv'
    with open(cad, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            a_aux.append(', '.join(row))
        a_3.append(float(a_aux[len(a_aux)-1:][0]))
        a_aux = []

arquitecturas = []
epochs_totales = []
incremento_epocas_array = []
mejora_aver_fit_array = []

# SI EL % DE VECES QUE LOS MEJORES INDIVIDUOS DE LA ITERACIÓN 1000 SON IGUALES ES ALTO, INDICA QUE DEJAR QUE LA POBLACIÓN TERMINE DE EJECUTARSE 
# MEJORA EL PROMEDIO DEL GRADO DE ADAPTACIÓN PERO NO EL MEJOR INDIVIDUO OBTENIDO, POR LO QUE NO INTERESA 
individuo_cambia = 0
for i in range(25):
    cad = cadd + str(i) + '.txt'
    with open(cad) as f:
        lines = f.readlines()

    tiempo = ''
    epochs = ''
    arquitectura = ''
    # Variables para manejar los programas con ECO (el txt varia)
    best_ind_1 = []
    best_ind_2 = []
    best_fit_1 = []
    best_fit_2 = []
    aver_fit_1 = []
    aver_fit_2 = []
    epoc_acc_1 = []
    epoc_acc_2 = []
    for j in range(len(lines)):
        if j == 0:
            k = len('Tiempo total ejecución: ')
            while lines[j][k] != '\n':
                tiempo += lines[j][k]
                k += 1
        elif j == 1:
            k = len('Épocas totales de entrenamiento: ')
            while lines[j][k] != '\n':
                epochs += lines[j][k]
                k += 1
        elif j == 2:
            k = len('Mejor individuo: ')
            while lines[j][k] != '\n':
                arquitectura += lines[j][k]
                k += 1

        if (execution == 1 or execution == 3) and j == 4:
            k = lines[j].find('Best individual: ')
            k += len('Best individual: ')
            while lines[j][k] != '.':
                best_ind_1 += lines[j][k]
                k += 1
            k = lines[j].find('Best fitness: ')
            k += len('Best fitness: ')
            while lines[j][k] != ' ':
                best_fit_1 += lines[j][k]
                k += 1
            best_fit_1 = best_fit_1[:len(best_fit_1)-1]
            k = lines[j].find('Population average: ')
            k += len('Population average: ')
            while lines[j][k] != ' ':
                aver_fit_1 += lines[j][k]
                k += 1
            k = lines[j].find('Epoch accum: ')
            k += len('Epoch accum: ')
            while lines[j][k] != '\n':
                epoc_acc_1 += lines[j][k]
                k += 1

        if (execution == 1 or execution == 3) and j == 5:
            k = lines[j].find('Best individual: ')
            k += len('Best individual: ')
            while lines[j][k] != '.':
                best_ind_2 += lines[j][k]
                k += 1
            k = lines[j].find('Best fitness: ')
            k += len('Best fitness: ')
            while lines[j][k] != ' ':
                best_fit_2 += lines[j][k]
                k += 1
            best_fit_2 = best_fit_2[:len(best_fit_2)-1]
            k = lines[j].find('Population average: ')
            k += len('Population average: ')
            while lines[j][k] != ' ':
                aver_fit_2 += lines[j][k]
                k += 1
            k = lines[j].find('Epoch accum: ')
            k += len('Epoch accum: ')
            while lines[j][k] != '\n':
                epoc_acc_2 += lines[j][k]
                k += 1
    
    
    if (execution == 1 or execution == 3):
        mejora_aver_fit = np.abs(float(convert(aver_fit_1)) - float(convert(aver_fit_2)))
        incremento_epocas = int(convert(epoc_acc_2)) - int(convert(epoc_acc_1))
        # Guardo el nº de veces que el mejor individuo cambia, es decir, interesa ejecutar el entrenamiento final
        if convert(best_ind_1) != convert(best_ind_2):
            individuo_cambia += 1
    
        mejora_aver_fit_array.append(mejora_aver_fit)
        incremento_epocas_array.append(incremento_epocas)
    epochs_totales.append(int(epochs))
    arquitecturas.append(arquitectura)


# Saco max, min, media y desviacion tipica
print("## Épocas de entrenamiento ##")
print( str(round(np.mean(epochs_totales),2)) + " & " + str(round(np.std(epochs_totales),2)) + " & " + str(np.max(epochs_totales)) + " & " + str(np.min(epochs_totales)))
print("")
print("## MSE/Precisión (Promedio) ##")
print( str(round(np.mean(a_1),5)) + " & " + str(round(np.std(a_1),5)) + " & " + str(round(np.max(a_1),5)) + " & " + str(round(np.min(a_1),5)))
print("")
print("## MSE/Precisión (TOP) ##")
print( str(round(np.mean(a_3),5)) + " & " + str(round(np.std(a_3),5)) + " & " + str(round(np.max(a_3),5)) + " & " + str(round(np.min(a_3),5)))
print("")
print("## MSE/Precisión (Mejor) ##")
print( str(round(np.mean(a_2),5)) + " & " + str(round(np.std(a_2),5)) + " & " + str(round(np.max(a_2),5)) + " & " + str(round(np.min(a_2),5)))
if (execution == 1 or execution == 3):
    print("")
    print("-------------- SISTEMAS CON ECO --------------")
    print("")
    print("## Variación del grado de adaptacion del promedio de la población tras dejar que la población se termine de entrenar ##")
    print( str(round(np.mean(mejora_aver_fit_array),5)) + " & " + str(round(np.std(mejora_aver_fit_array),5)) + " & " + str(round(np.max(mejora_aver_fit_array),5)) + " & " + str(round(np.min(mejora_aver_fit_array),5)))
    print("")
    print("## Incremento de épocas tras dejar que la población se termine de entrenar ##")
    print( str(round(np.mean(incremento_epocas_array),5)) + " & " + str(round(np.std(incremento_epocas_array),5)) + " & " + str(round(np.max(incremento_epocas_array),5)) + " & " + str(round(np.min(incremento_epocas_array),5)))
    print("")
    print("Porcentaje de variación del mejor individuo (sistemas con ECO): " + str(individuo_cambia) + "%")
    print("----------------------------------------------")

print("")
print("--- Arquitecturas ---")
# Obtengo la topologia de cada arquitectura (Por ejemplo, el individuo 11011111 es el 2:5)
topologias = []
for i in range(len(arquitecturas)):
    num_neuronas = 0
    new_arq = ''
    for j in range(len(arquitecturas[i])):
        if arquitecturas[i][j] == '1': num_neuronas += 1
        else: 
            new_arq += str(num_neuronas) + ':'
            num_neuronas = 0
    new_arq += str(num_neuronas) + ':'
    new_arq = new_arq[:-1]
    topologias.append(new_arq)
#print(topologias)
# Obtengo el número de capas ocultas que tiene cada individuo
num_capas_array = []
for i in range(len(topologias)):
    n_capas = 0
    for j in range(len(topologias[i])):
        if topologias[i][j] == ":":
            n_capas += 1
    n_capas += 1
    num_capas_array.append(n_capas)
#print(num_capas_array)
# Calculo cuantos individuos hay de cada tipo (siendo un tipo topologia de 2 capas, de 3, etc)
n_capas_max = np.max(num_capas_array)
n_ind_array = []
for i in range(2, n_capas_max+1):
    n_ind = 0
    for j in range(len(num_capas_array)):
        if num_capas_array[j] == i:
            n_ind += 1
    n_ind_array.append(n_ind)
for i in range(len(n_ind_array)):
    print("Hay " + str(n_ind_array[i]) + " individuos con " + str(i+2) + " capas ocultas.")

max_ind = 0
max_value = n_ind_array[0]
for i in range(1, len(n_ind_array)):
    if n_ind_array[i] > max_value:
        max_value = n_ind_array[i]
        max_ind = i

print("Analizamos en profundidad los individuos con " + str(max_ind+2) + " capas ocultas (interesan porque hay más)")
n_capas_interesan = max_ind+2

topologias_interesan = []
for i in range(len(num_capas_array)):
    if num_capas_array[i] == n_capas_interesan:
        topologias_interesan.append(topologias[i])
#print(topologias_interesan)


def get_neuronas_capa_i(topologia, capa):
    n_neuronas_array = []
    cad = ''
    for i in range(len(topologia)):
        if topologia[i] != ":":
            cad += topologia[i]
        else:
            n_neuronas_array.append(int(cad))
            cad = ''
    n_neuronas_array.append(int(cad))
    return n_neuronas_array[capa]


for i in range(n_capas_interesan):
    n_neuronas_array = []
    for j in range(len(topologias_interesan)):
        n_neuronas_array.append(get_neuronas_capa_i(topologias_interesan[j], i))
    print("")
    print("## Distribución neuronas de la capa oculta " + str(i) + " ##")
    print( str(round(np.mean(n_neuronas_array),5)) + " & " + str(round(np.std(n_neuronas_array),5)) + " & " + str(round(np.max(n_neuronas_array),5)) + " & " + str(round(np.min(n_neuronas_array),5)))