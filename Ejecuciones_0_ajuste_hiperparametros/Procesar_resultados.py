cad1 = './Resultados/AsynCali.txt'
cad2 = './Resultados/AsynFIFA.txt'
cad3 = './Resultados/SynCali.txt'
cad4 = './Resultados/SynFIFA.txt'

# Cambiar esto
entrada = cad1

with open(entrada) as f:
    lines = f.readlines()

best_f = []
aveg_f = []
l1 = len('Best fitness: ')
l2 = len('Average fitness: ')
for i in range(len(lines)):
    result1 = lines[i].find('Best fitness: ')
    result2 = lines[i].find('Average fitness: ')
    n1 = lines[i][result1+l1] + lines[i][result1+l1+1]
    j = result1+l1+2
    while lines[i][j] != '.':
        n1 += lines[i][j]
        j += 1
    n2 = ''
    j = result2+l2
    while lines[i][j] != ' ':
        n2 += lines[i][j]
        j += 1
    best_f.append(float(n1))
    aveg_f.append(float(n2))
    

best_f_ind = 0
best_f_num = best_f[best_f_ind]
aveg_f_ind = 0
aveg_f_num = aveg_f[aveg_f_ind]

for i in range(1, len(best_f)):
    if "FIFA" in entrada: #MAX
        if best_f[i] > best_f_num:
            best_f_ind = i
            best_f_num = best_f[best_f_ind]
    else: #MIN
        if best_f[i] < best_f_num:
            best_f_ind = i
            best_f_num = best_f[best_f_ind]

for i in range(1, len(aveg_f)):
    if "FIFA" in entrada: #MAX
        if aveg_f[i] > aveg_f_num:
            aveg_f_ind = i
            aveg_f_num = aveg_f[aveg_f_ind]
    else: #MIN
        if aveg_f[i] < aveg_f_num:
            aveg_f_ind = i
            aveg_f_num = aveg_f[aveg_f_ind]

print("best_f. Ind: " + str(best_f_ind+1) + "; value: " + str(best_f_num))
print("aveg_f. Ind: " + str(aveg_f_ind+1) + "; value: " + str(aveg_f_num))