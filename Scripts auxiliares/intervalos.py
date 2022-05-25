"""
array_1 = Sistema secuencial (sin ECO)
array_2 = Sistema concurrente (con ECO)

Cada array tiene en [0] el máximo y en [1] el mínimo
"""
for execution in range(4):
    if execution == 0:
        array_1 = [134102, 28482]
        array_2 = [43837, 5992]
        print("Épocas BP FIFA")
    elif execution == 1:
        array_1 = [0.82568, 0.81079]
        array_2 = [0.83437, 0.82196]
        print("Best Ind. FIFA")
    elif execution == 2:
        array_1 = [196139, 16422]
        array_2 = [90141, 17623]
        print("Épocas BP California")
    elif execution == 3:
        array_1 = [0.21973, 0.21532]
        array_2 = [0.22064, 0.217]
        print("Best Ind. California")

    # Porcentaje solapamiento de dos intervalos = (interseccion/union)*100

    union_min = min(array_1[1], array_2[1])
    union_max = max(array_1[0], array_2[0])
    long_union = union_max - union_min

    inter_min = max(array_1[1], array_2[1])
    inter_max = min(array_1[0], array_2[0])
    long_inter = inter_max - inter_min

    porcentaje_solapamiento = (long_inter / long_union) * 100
    print("El porcentaje de solapamiento de los intervalos es: " + str(porcentaje_solapamiento) + "%.")
    print()
    print()