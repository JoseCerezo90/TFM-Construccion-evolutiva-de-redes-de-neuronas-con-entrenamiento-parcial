import pandas as pd

dataset = pd.read_csv("./iris_raw.csv", sep=",")

"""print(dataset['variety'])

Setosa_array = []
Versicolor_array = []
Virginica_array = []
for i in range(150):
    if dataset['variety'][i] == "Setosa":
        Setosa_array += [1.0] 
        Versicolor_array += [0.0]
        Virginica_array += [0.0]
    elif dataset['variety'][i] == "Versicolor":
        Setosa_array += [0.0] 
        Versicolor_array += [1.0]
        Virginica_array += [0.0]
    elif dataset['variety'][i] == "Virginica":
        Setosa_array += [0.0] 
        Versicolor_array += [0.0]
        Virginica_array += [1.0]


data1 = {'Setosa':  Setosa_array,
        'Versicolor': Versicolor_array,
        'Virginica': Virginica_array
        }

df = pd.DataFrame(data1)
df.to_csv('./Limpio3.csv', index = False)"""

def get_comas(cadena):
    sp = ''
    sw = ''
    pl = ''
    pw = ''
    var = ''
    num_comas = 0
    for i in range(len(cadena)):
        if cadena[i] == ',':
            num_comas+=1
        else:
            if num_comas==0:
                sp += cadena[i]
            elif num_comas==1:
                sw += cadena[i]
            elif num_comas==2:
                pl += cadena[i]
            elif num_comas==3:
                pw += cadena[i]
            elif num_comas==4 and cadena[i]!='"':
                var += cadena[i]
    return sp, sw, pl, pw, var 

sp_array = []
sw_array = []
pl_array = []
pw_array = []
var_array = []
for i in range(150):
    sp, sw, pl, pw, var = get_comas(dataset['sepal.length'][i])
    sp_array += [float(sp)]
    sw_array += [float(sw)]
    pl_array += [float(pl)]
    pw_array += [float(pw)]
    var_array += [var]


data1 = {'sepal.length':  sp_array,
        'sepal.width': sw_array,
        'petal.length': pl_array,
        'petal.width': pw_array
        }



df = pd.DataFrame(data1)
df.to_csv('./Limpiooo.csv', index=False)