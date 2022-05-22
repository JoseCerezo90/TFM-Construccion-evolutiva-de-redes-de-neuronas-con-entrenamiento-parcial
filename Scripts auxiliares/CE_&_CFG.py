import random
import numpy as np
from random import randrange

#####################
# GENERAR INDIVIDUO #
#####################

# Un individuo se representa mediante una lista de tuplas, en la cual, la primera componente es la regla que se aplica
# sobre el terminal que se encuentre más a la izquierda en el árbol de derivación actual y la segunda componente corresponderá
# a la altura a la que se encuentra este no terminal en el árbol (esto nos será útil en el cruce)
def generarIndividuoAleatorio(raiz, profundidad_inicial):
    individuo = []
    i = 0
    arbol = raiz
    profundidad = profundidad_inicial
    no_terminal = buscarSiguienteNoTerminal(arbol)
    pila_profundidades = []
    while i < 60:
        n_reglas = len(gramatica.get(no_terminal))
        regla = randrange(n_reglas)
        individuo.append((regla, profundidad))
        profundidad += 1
        new_regla = gramatica.get(no_terminal)[regla]
        if len(new_regla) > 1:
            pila_profundidades.append(profundidad)
        if len(pila_profundidades) > 0 and (new_regla == '0' or new_regla == '1' or new_regla == '2' or new_regla == '3' or new_regla == '4' or \
            new_regla == '5' or new_regla == '6' or new_regla == '7' or new_regla == '8' or new_regla == '9'):
            profundidad = pila_profundidades.pop()
        arbol = sustituirRegla(arbol, new_regla, no_terminal)
        no_terminal = buscarSiguienteNoTerminal(arbol)
        i += 1
        if no_terminal == '':
            return individuo
    return []
    
##################################
# DECODIFICACIÓN DE UN INDIVIDUO #
##################################

def buscarSiguienteNoTerminal(regla):
    regla = [caracter for caracter in regla]
    for r in regla:
        if r in no_terminales:
            return r
    return ''

def sustituirRegla(arbol, regla, no_terminal): 
    new_arbol = ''
    sustituido = False
    for i in range(len(arbol)):
        if arbol[i] == no_terminal and not sustituido:
            new_arbol += regla
            sustituido = True
        else:
            new_arbol += arbol[i]
    return new_arbol

def decodificarIndividuo(individuo):
    arbol = 'S'
    no_terminal = buscarSiguienteNoTerminal(arbol)
    for (regla, _) in individuo:
        new_regla = gramatica.get(no_terminal)[regla]
        arbol = sustituirRegla(arbol, new_regla, no_terminal)
        no_terminal = buscarSiguienteNoTerminal(arbol)
        if no_terminal == '':
            return arbol

#########################
# OPERADORES EVOLUTIVOS #
#########################

def generarPoblacionInicial(tamPoblacion):
    poblacion = []
    while len(poblacion) < tamPoblacion:
        individuo = generarIndividuoAleatorio('S', 0)
        if individuo != []: poblacion.append(individuo)
    return poblacion

# Función fitness, diferencia absoluta entre la parte izquierda y derecha de la igualdad
def evaluarIndividuo(individuo):
    individuo = decodificarIndividuo(individuo)
    i = 0 
    parteIzq = ''
    while individuo[i][0] != '=':
        parteIzq += individuo[i][0]
        i += 1
    parteDer = individuo[i+1][0]
    parteIzq = [caracter for caracter in parteIzq]
    accum = int(parteIzq[0])
    for i in range(1, len(parteIzq)):
        if parteIzq[i] == '+':
            accum += int(parteIzq[i+1])
        elif parteIzq[i] == '-':
            accum -= int(parteIzq[i+1])
        i += 2
    return np.abs(np.abs(accum)-int(parteDer))

# Método de la ruleta (menor fitness implica mejor individuo)
def Seleccion(poblacion, individuos_a_seleccionar):
    seleccionados = []
    fitness = []
    # Calculo las probabilidades
    for i in range(len(poblacion)):
        fit = evaluarIndividuo(poblacion[i])
        fitness.append(1/(fit+1))
    sum_fitness = 0
    for i in range(len(poblacion)):
        sum_fitness += fitness[i]
    for i in range(len(poblacion)):
        fitness[i] = fitness[i] / sum_fitness
    # Hago las particiones de la ruleta 
    probs = []
    for i in range(individuos_a_seleccionar):
        probs.append((i+1)/(individuos_a_seleccionar+1))
    # Me quedo con los individuos
    accum_fitness = 0
    accum_fitness_pasado = 0
    for i in range(len(poblacion)):
        accum_fitness += fitness[i]
        for j in range(len(probs)):
            if accum_fitness >= probs[j] and accum_fitness_pasado < probs[j]:
                seleccionados.append(poblacion[i])
        accum_fitness_pasado += fitness[i]
    return seleccionados

# Function used in the crossover and mutation function.
# Extracts the set of all leads of an individual and its nonterminals
# Extrae el conjunto de todas las derivaciones de un individuo y sus no terminales
def obtenerNodosNoTerminales(individuo):
    arbol_completo = []
    no_terminales = []
    # Obtengo el árbol completo de un individuo (ej: ['E=N', 'F+E=N', 'N+E=N', '7+E=N', '7+N=N', '7+9=N', '7+9=5'])
    # y sus nodos no terminales (ej: [('E', 2), ('F', 0), ('N', 7), ('E', 4), ('N', 9), ('N', 5)])
    arbol = 'S'
    no_terminal = buscarSiguienteNoTerminal(arbol)
    for i in range(len(individuo)):
        new_regla = gramatica.get(no_terminal)[individuo[i][0]]
        arbol = sustituirRegla(arbol, new_regla, no_terminal)
        arbol_completo.append(arbol)
        no_terminal = buscarSiguienteNoTerminal(arbol)
        if (i+1) < len(individuo):
            no_terminales.append((no_terminal, individuo[i+1][0]))
    return arbol_completo, no_terminales

# Función empleada en la función de cruce
# Extrae el conjunto de símbolos no terminales que se pueden cruzar
def nodosDeCruce(no_terminales1, no_terminales2):
    no_t_1 = []
    no_t_2 = []
    for i in range(len(no_terminales1)):
        no_t_1.append(no_terminales1[i][0])
    for i in range(len(no_terminales2)):
        no_t_2.append(no_terminales2[i][0])
    no_t_1 = set(no_t_1)
    no_t_2 = set(no_t_2)
    # Retorno la intersección de los no terminales (que se podrán cruzar)
    return no_t_1.intersection(no_t_2)

def buscarLugarCruce(simbolo_a_cruzar, no_terminales):
    lugar_cruce = -1
    while lugar_cruce == -1:
        ind = randrange(len(no_terminales))
        if no_terminales[ind][0] == simbolo_a_cruzar:
            lugar_cruce = ind
            return lugar_cruce

def obtenerSubarbol(padre, no_terminales, lugar_padre):
    padre = recalcularProfundidades(padre)
    fin_subarbol = 0
    # Lugar de inicio del subárbol en la cadena raw (individuo original)
    inicio_subarbol = lugar_padre + 1
    if no_terminales[lugar_padre][0] == 'N':
        fin_subarbol = inicio_subarbol
    elif no_terminales[lugar_padre][0] == 'F':
        fin_subarbol = inicio_subarbol+1
    else:
        prof_ini = padre[lugar_padre+1][1] - 1
        flag = True
        for i in range(lugar_padre+2, len(padre)):
            if flag and prof_ini >= (padre[i][1]-1):
                fin_subarbol = i - 1
                flag = False
    return inicio_subarbol, fin_subarbol

"""""""""""""""""""""""""""""""""""""""""""""""""""
#                                                 #
#                                                 #
#    TENGO HECHO HASTA AQUÍ EN EL CÓDIGO BUENO    #
#                                                 #
#                                                 #
"""""""""""""""""""""""""""""""""""""""""""""""""""

def recalcularProfundidades(individuo):
    new_individuo = []
    i = 0
    arbol = 'S'
    profundidad = 0
    no_terminal = buscarSiguienteNoTerminal(arbol)
    pila_profundidades = []
    for i in range(len(individuo)):
        regla = individuo[i][0]
        new_individuo.append((regla, profundidad))
        profundidad += 1
        new_regla = gramatica.get(no_terminal)[regla]
        if len(new_regla) > 1:
            pila_profundidades.append(profundidad)
        if len(pila_profundidades) > 0 and (new_regla == '0' or new_regla == '1' or new_regla == '2' or new_regla == '3' or new_regla == '4' or \
            new_regla == '5' or new_regla == '6' or new_regla == '7' or new_regla == '8' or new_regla == '9'):
            profundidad = pila_profundidades.pop()
        arbol = sustituirRegla(arbol, new_regla, no_terminal)
        no_terminal = buscarSiguienteNoTerminal(arbol)
        i += 1
    return new_individuo

def intercambiarSubarboles(padre1, padre2, inicio_subarbol1, fin_subarbol1, inicio_subarbol2, fin_subarbol2):
    subarbol1 = padre1[inicio_subarbol1:fin_subarbol1+1]
    subarbol2 = padre2[inicio_subarbol2:fin_subarbol2+1]
    hijo1 = padre1[:inicio_subarbol1] + subarbol2 + padre1[fin_subarbol1+1:]
    hijo2 = padre2[:inicio_subarbol2] + subarbol1 + padre2[fin_subarbol2+1:]
    return recalcularProfundidades(hijo1), recalcularProfundidades(hijo2)

# Cruce Whigham (WX)
def Cruce(padre1, padre2):
    arbol_completo1, no_terminales1 = obtenerNodosNoTerminales(padre1)
    arbol_completo2, no_terminales2 = obtenerNodosNoTerminales(padre2)
    # Obtengo los nodos de cruce candidatos para Whigham
    nodos_cruce = nodosDeCruce(no_terminales1, no_terminales2)
    # Si la intersección está vacía retorno hijos nulos
    if nodos_cruce == set(): return [], []
    # Del conjunto intersección, extraigo aleatoriamente un símbolo no terminal de cruce
    simbolo_a_cruzar = list(nodos_cruce)[randrange(len(nodos_cruce))]
    # Busco este símbolo en el padre1 (si hay varias me quedo con uno aleatorio) 
    lugar_padre1 = buscarLugarCruce(simbolo_a_cruzar, no_terminales1) 
    lugar_padre2 = buscarLugarCruce(simbolo_a_cruzar, no_terminales2)
    # Veo dónde empieza y acaba cada subárbol del nodo de cruce en cada padre
    inicio_subarbol1, fin_subarbol1 = obtenerSubarbol(padre1, no_terminales1, lugar_padre1)
    inicio_subarbol2, fin_subarbol2 = obtenerSubarbol(padre2, no_terminales2, lugar_padre2)
    # Realizo el cruce
    hijo1, hijo2 = intercambiarSubarboles(padre1, padre2, inicio_subarbol1, fin_subarbol1, inicio_subarbol2, fin_subarbol2)
    return hijo1, hijo2

# Mutación por fuerza bruta (aleatorio)
def Mutacion(poblacion, prob_mutacion):
    new_poblacion = []
    for i in range(len(poblacion)):
        # No se muta el individuo
        if random.uniform(0, 1) > prob_mutacion:
            new_poblacion.append(poblacion[i])
        # Muta
        else:
            # Busco un nodo sobre el que realizar la mutación (aleatorio)
            arbol_completo, no_terminales = obtenerNodosNoTerminales(poblacion[i])
            posicion_nodo_a_mutar = randrange(len(poblacion[i])-1)
            # Obtengo su subárbol
            inicio_subarbol, fin_subarbol = obtenerSubarbol(poblacion[i], no_terminales, posicion_nodo_a_mutar)
            # Genero el subárbol a partir del nodo a mutar
            subarbol_mutado = []
            while subarbol_mutado == []:
                subarbol_mutado = generarIndividuoAleatorio(no_terminales[posicion_nodo_a_mutar][0], poblacion[i][posicion_nodo_a_mutar][1]+1)
            # Genero el individuo mutado
            mutado = poblacion[i][:inicio_subarbol] + subarbol_mutado + poblacion[i][fin_subarbol+1:]
            new_poblacion.append(mutado)
    return new_poblacion

# Reemplazo estacionado (SSGA)
def Reemplazo(poblacion, hijos):
    poblacion_eval = []
    for i in range(len(poblacion)):
        poblacion_eval.append((evaluarIndividuo(poblacion[i]), poblacion[i]))
    # Ordeno a la poblacion en función del fitness (orden decreciente)
    sorted_poblacion_eval = sorted(poblacion_eval, key=lambda x: x[0], reverse=True)
    # Elimino de la población los número_de_hijos peores
    poblacion_recortada = sorted_poblacion_eval[:(len(poblacion)-len(hijos))]
    # La nueva población es la que queda + los hijos
    new_poblacion = []
    for i in range(len(poblacion_recortada)):
        new_poblacion.append(poblacion_recortada[i][1])
    for i in range(len(hijos)):
        new_poblacion.append(hijos[i])
    return new_poblacion

# Condición de parada: Se supera el máximo de iteraciones o toda la población ha convergido (fitness igual a 0)
def condicionParada(poblacion, iteracion):
    num_it_max = 1000
    if iteracion > num_it_max:
        return True
    for i in range(len(poblacion)):
        if evaluarIndividuo(poblacion[i]) != 0:
            return False
    return True

########
# MAIN #
########

if __name__ == "__main__":
    # Gramática
    no_terminales = ['S','E','F','N']
    gramatica = {
        'S':['E=N'],
        'E':['E+E','E-E','F+E','F-E','N'],
        'F':['N'],
        'N':['0','1','2','3','4','5','6','7','8','9']
    }
    # Parámetros
    tamPoblacion = 200
    individuos_a_seleccionar = 160
    prob_cruce = 0.95
    prob_mutacion = 0.05
    # Genero la población inicial
    poblacion = generarPoblacionInicial(tamPoblacion)
    iteracion = 0
    while not condicionParada(poblacion, iteracion):
        iteracion += 1
        hijos = []
        # Selección
        seleccionados = Seleccion(poblacion, individuos_a_seleccionar)
        # Cruce 
        for i in range(int(len(seleccionados)/2)):
            r = random.uniform(0, 1)
            if r <= prob_cruce:
                # Busco los 2 padres entre los seleccionados
                indice_padre1 = randrange(len(seleccionados))
                padre1 = seleccionados[indice_padre1]
                # Elimino los seleccionados para que no se repitan
                seleccionados.pop(indice_padre1)
                indice_padre2 = randrange(len(seleccionados))
                padre2 = seleccionados[indice_padre2]
                seleccionados.pop(indice_padre2)
                # Ahora ya tengo a los dos padres a cruzar
                hijo1, hijo2 = Cruce(padre1, padre2)
                if hijo1 != [] and hijo2 != []:
                    hijos.append(hijo1)
                    hijos.append(hijo2)
        # Mutación 
        poblacion = Mutacion(poblacion, prob_mutacion)
        # Reemplazo
        poblacion = Reemplazo(poblacion, hijos)
    # Muestro la población resultante del proceso evolutivo
    for i in range(len(poblacion)):
        print(str(i+1) + "º individuo: " + str(poblacion[i]) + "  <->  " + str(decodificarIndividuo(poblacion[i])) + \
            "  <->  Fitness: " + str(evaluarIndividuo(poblacion[i])))