import csv
import pandas as pd
import numpy as np

base_chain = "../Ejecuciones_3_libre_albedrio/Ejecuciones_libre_albedrio/"

for execution in range(4):

    if execution == 0:
        cad1 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Aver_Fit_'
        cad2 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Best_Fit_'
        cad3 = base_chain + 'California/Asincrono/MedianHouseValue_Async_Epochs_'
        cad4 = base_chain + 'California/Asincrono/MedianHouseValue_Async_TOP_Fit_'
    elif execution == 1:
        cad1 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Aver_Fit_'
        cad2 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Best_Fit_'
        cad3 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_Epochs_'
        cad4 = base_chain + 'California/Sincrono/MedianHouseValue_Sync_TOP_Fit_'
    elif execution == 2:
        cad1 = base_chain + 'FIFA/Asincrono/FIFA_Async_Aver_Fit_'
        cad2 = base_chain + 'FIFA/Asincrono/FIFA_Async_Best_Fit_'
        cad3 = base_chain + 'FIFA/Asincrono/FIFA_Async_Epochs_'
        cad4 = base_chain + 'FIFA/Asincrono/FIFA_Async_TOP_Fit_'
    else:
        cad1 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Aver_Fit_'
        cad2 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Best_Fit_'
        cad3 = base_chain + 'FIFA/Sincrono/FIFA_Sync_Epochs_'
        cad4 = base_chain + 'FIFA/Sincrono/FIFA_Sync_TOP_Fit_'

    # Calculate what has been the longest evolutionary process (greatest number of iterations)
    max_length = 0
    for i in range(25):
        input_file = open(cad1 + str(i) + '.csv',"r+")
        reader_file = csv.reader(input_file)
        value = len(list(reader_file))
        if value > max_length:
            max_length = value
            
    # 25 executions per problem and system
    array1 = np.zeros((25, max_length))
    array2 = np.zeros((25, max_length))
    array3 = np.zeros((25, max_length))
    array4 = np.zeros((25, max_length))

    # Read all the files
    for n_cad in range(4):
        for i in range(25):
            if n_cad == 0: cad = cad1 + str(i) + '.csv'
            elif n_cad == 1: cad = cad2 + str(i) + '.csv'
            elif n_cad == 2: cad = cad3 + str(i) + '.csv'
            elif n_cad == 3: cad = cad4 + str(i) + '.csv'
            with open(cad, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                j = 0
                for row in spamreader:
                    if n_cad == 0: array1[i][j] = ', '.join(row)
                    elif n_cad == 1: array2[i][j] = ', '.join(row)
                    elif n_cad == 2: array3[i][j] = ', '.join(row)
                    elif n_cad == 3: array4[i][j] = ', '.join(row)
                    j += 1

    # Calculate the average for each file
    new_array1 = []
    new_array2 = []
    new_array3 = []
    new_array4 = []
    for n_cad in range(4):
        for i in range(max_length):
            accum = 0
            cont = 0
            for j in range(25):
                if n_cad == 0: 
                    accum += array1[j][i]
                    if array1[j][i] != 0: cont += 1
                elif n_cad == 1: 
                    accum += array2[j][i]
                    if array2[j][i] != 0: cont += 1
                elif n_cad == 2: 
                    accum += array3[j][i]
                    if array3[j][i] != 0: cont += 1
                elif n_cad == 3: 
                    accum += array4[j][i]
                    if array4[j][i] != 0: cont += 1
            accum /= cont

            if n_cad == 0: new_array1.append(accum)
            elif n_cad == 1: new_array2.append(accum)
            elif n_cad == 2: new_array3.append(accum)
            elif n_cad == 3: new_array4.append(accum)
            
    df1 = pd.DataFrame(new_array1)
    df2 = pd.DataFrame(new_array2)
    df3 = pd.DataFrame(new_array3)
    df4 = pd.DataFrame(new_array4)

    if execution == 0:
        df1.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Async_Aver_Fit.csv', index=False, header=False)
        df2.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Async_Best_Fit.csv', index=False, header=False)
        df3.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Async_Epochs.csv', index=False, header=False)
        df4.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Async_TOP_Fit.csv', index=False, header=False)
    elif execution == 1:
        df1.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Sync_Aver_Fit.csv', index=False, header=False)
        df2.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Sync_Best_Fit.csv', index=False, header=False)
        df3.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Sync_Epochs.csv', index=False, header=False)
        df4.to_csv(base_chain + 'Resultados_promediados/MedianHouseValue_Sync_TOP_Fit.csv', index=False, header=False)
    elif execution == 2:
        df1.to_csv(base_chain + 'Resultados_promediados/FIFA_Async_Aver_Fit.csv', index=False, header=False)
        df2.to_csv(base_chain + 'Resultados_promediados/FIFA_Async_Best_Fit.csv', index=False, header=False)
        df3.to_csv(base_chain + 'Resultados_promediados/FIFA_Async_Epochs.csv', index=False, header=False)
        df4.to_csv(base_chain + 'Resultados_promediados/FIFA_Async_TOP_Fit.csv', index=False, header=False)
    else:
        df1.to_csv(base_chain + 'Resultados_promediados/FIFA_Sync_Aver_Fit.csv', index=False, header=False)
        df2.to_csv(base_chain + 'Resultados_promediados/FIFA_Sync_Best_Fit.csv', index=False, header=False)
        df3.to_csv(base_chain + 'Resultados_promediados/FIFA_Sync_Epochs.csv', index=False, header=False)
        df4.to_csv(base_chain + 'Resultados_promediados/FIFA_Sync_TOP_Fit.csv', index=False, header=False)