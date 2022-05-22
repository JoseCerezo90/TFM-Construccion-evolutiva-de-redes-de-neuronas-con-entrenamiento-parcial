import csv
import pandas as pd
import numpy as np
from scipy import stats


for execution in range(4):

    # 100 ficheros. Para cada fichero busco cuándo se alcanza la solución (cota: FIFA > 0.825; California < 0.23) y guardo esa iteración y número de épocas BP
    array = np.zeros((100, 2))
    arrayy = np.zeros((100, 2))

    if execution == 0:
        cad1 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/California/Síncrono/MedianHouseValue_Sync_Best_Fit_'
        cad2 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/California/Síncrono/MedianHouseValue_Sync_Epochs_'
        cad3 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/California/Asíncrono/MedianHouseValue_Async_Best_Fit_'
        cad4 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/California/Asíncrono/MedianHouseValue_Async_Epochs_'
    elif execution == 1:
        cad1 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/California/Sincrono/MedianHouseValue_Sync_Best_Fit_'
        cad2 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/California/Sincrono/MedianHouseValue_Sync_Epochs_'
        cad3 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/California/Asincrono/MedianHouseValue_Async_Best_Fit_'
        cad4 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/California/Asincrono/MedianHouseValue_Async_Epochs_'
    elif execution == 2:
        cad1 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/FIFA/Síncrono/FIFA_Sync_Best_Fit_'
        cad2 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/FIFA/Síncrono/FIFA_Sync_Epochs_'
        cad3 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/FIFA/Asíncrono/FIFA_Async_Best_Fit_'
        cad4 = '../Ejecuciones_2_acotadas/Ejecuciones_CON_GUPI/FIFA/Asíncrono/FIFA_Async_Epochs_'
    else:
        cad1 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/FIFA/Sincrono/FIFA_Sync_Best_Fit_'
        cad2 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/FIFA/Sincrono/FIFA_Sync_Epochs_'
        cad3 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/FIFA/Asincrono/FIFA_Async_Best_Fit_'
        cad4 = '../Ejecuciones_2_acotadas/Ejecuciones_SIN_GUPI/FIFA/Asincrono/FIFA_Async_Epochs_'

    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    for i in range(100):
        # Leo ambos archivos
        cad = cad1 + str(i) + '.csv'
        with open(cad, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                array_1.append(float(', '.join(row)))
        cad = cad2 + str(i) + '.csv'
        with open(cad, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                array_2.append(float(', '.join(row)))
        # Veo cuando se alcanza la cota
        flag = True
        j = 0
        while flag and j < len(array_1):
            if execution == 0 or execution == 1:
                if array_1[j] < 0.23:
                    array[i][0] = j
                    flag = False
            else:
                if array_1[j] > 0.825:
                    array[i][0] = j
                    flag = False
            j += 1
        array[i][1] = array_2[j]
        array_1 = []
        array_2 = []
        # Leo ambos archivos
        cad = cad3 + str(i) + '.csv'
        with open(cad, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                array_3.append(float(', '.join(row)))
        cad = cad4 + str(i) + '.csv'
        with open(cad, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                array_4.append(float(', '.join(row)))
        # Veo cuando se alcanza la cota
        flag = True
        j = 0
        while flag and j < len(array_3):
            if execution == 0 or execution == 1:
                if array_3[j] < 0.23:
                    arrayy[i][0] = j
                    flag = False
            else:
                if array_3[j] > 0.825:
                    arrayy[i][0] = j
                    flag = False
            j += 1
        arrayy[i][1] = array_4[j]
        array_3 = []
        array_4 = []

        
    array_iteraciones_1 = []
    array_epocas_BP_1 = []
    array_iteraciones_2 = []
    array_epocas_BP_2 = []
    for i in range(100):
        array_iteraciones_1.append(array[i][0])
        array_epocas_BP_1.append(array[i][1])
        array_iteraciones_2.append(arrayy[i][0])
        array_epocas_BP_2.append(arrayy[i][1])

    print("---- Sistema sin ECO (secuencial) ----")
    print("Iteración en la que se detiene (media, desv. típ., máx., mín.)")
    print( str(round(np.mean(array_iteraciones_1),5)) + " & " + str(round(np.std(array_iteraciones_1),5)) + " & " + str(round(np.max(array_iteraciones_1),5)) + " & " + str(round(np.min(array_iteraciones_1),5)))
    print("Épocas BP empleadas (media, desv. típ., máx., mín.)")
    print( str(round(np.mean(array_epocas_BP_1),5)) + " & " + str(round(np.std(array_epocas_BP_1),5)) + " & " + str(round(np.max(array_epocas_BP_1),5)) + " & " + str(round(np.min(array_epocas_BP_1),5)))
    print("")
    print("---- Sistema con ECO (concurrente) ----")
    print("Iteración en la que se detiene (media, desv. típ., máx., mín.)")
    print( str(round(np.mean(array_iteraciones_2),5)) + " & " + str(round(np.std(array_iteraciones_2),5)) + " & " + str(round(np.max(array_iteraciones_2),5)) + " & " + str(round(np.min(array_iteraciones_2),5)))
    print("Épocas BP empleadas (media, desv. típ., máx., mín.)")
    print( str(round(np.mean(array_epocas_BP_2),5)) + " & " + str(round(np.std(array_epocas_BP_2),5)) + " & " + str(round(np.max(array_epocas_BP_2),5)) + " & " + str(round(np.min(array_epocas_BP_2),5)))

    print()
    print()
    print()
    print("###############################")
    print("# Shapiro-Wilk test normality #")
    print("###############################")
    print(stats.shapiro(array_epocas_BP_1))
    print(stats.shapiro(array_epocas_BP_2))
    print()
    print("* Puesto que todos los p-value son < 0.05 se puede afirmar (con probabilidad de equivocarse del 5%) de que los datos no siguen una distribución normal")
    print("------------------------------------------------------------------------------------------------------------------------")
    print("#####################################################")
    print("# Test de Levene para comprobar la homocedasticidad #")
    print("#####################################################")
    print(stats.levene(array_epocas_BP_1, array_epocas_BP_2))
    print()
    print("* Puesto que el p-value es < 0.05, hay poca significancia, por lo que es poco probable que las varianzas sean similares, por lo que se rechaza la hipótesis nula y se concluye que hay diferencia entre las varianzas de las poblaciones.")
    print("------------------------------------------------------------------------------------------------------------------------")
    print("################")
    print("# Welch's test #")
    print("################")
    print(stats.ttest_ind(array_epocas_BP_1, array_epocas_BP_2, equal_var = False))
    print()
    print("* Puesto que el p-value es < 0.05, podemos rechazar la hipótesis nula de la prueba y concluir que la diferencia entre la media de ambos tipos es bastante significativa.")
    print()
    print()
    print()


    ############# IMPORTANTE: LA Prueba t de dos muestras asume que las varianzas entre los dos grupos son iguales --> COMO NO ES ASI SE USA WELCH, que es el equivalente no paramétrico de la prueba t de dos muestras.