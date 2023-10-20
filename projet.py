# -*- coding: utf-8 -*-

from random import *
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import copy

# Fonction qui lit un fichier contentant un graphe
def read_file(file_path):
    file = open(file_path, "r")
    lignes = file.readlines()
    nb_sommets = 0
    sommets = []
    nb_aretes = 0
    aretes = []
    flag = 0
    for ligne in lignes:
        if ligne == "Nombre de sommets\n":
            flag = 1
            continue
        if ligne == "Sommets\n":
            flag = 2
            continue
        if ligne == "Nombre d aretes\n":
            flag = 3
            continue
        if ligne == "Aretes\n":
            flag = 4
            continue
        else:
            if flag == 1:
                nb_sommets = int(ligne)
            if flag == 2:
                sommets.append(int(ligne))
            if flag == 3:
                nb_aretes = int(ligne)
            if flag == 4:
                aretes.append(ligne.split())
    return (nb_sommets, sommets, nb_aretes, aretes)

# Fonction de creation d'un graphe
def create_graph(nb_sommets, sommets, nb_aretes, aretes):
    G = dict()
    for sommet in sommets:
        G[str(sommet)] = []
    for arete in aretes:
        G[arete[0]].append(arete[1])
        G[arete[1]].append(arete[0])
    return G

# Fonction pour supprimer un sommet v d'un graphe
def delete_sommet(G, v):
    if v in G:
        del G[v]
        for l_value in G.values():
            if v in l_value:
                l_value.remove(v)
    return G

# Fonction qui supprime une liste de sommet d'un graphe
def delete_liste_sommets(G, lv):
    for v in lv:
        G = delete_sommet(G, v)
    return G

# Fonction qui retourne les degrés d'un graphe
def degre(G):
    return [len(value) for value in G.values()]

# Fonction qui retourne le ssommet ayant le degré maximum
def degre_max(G):
    value_max = max(degre(G))
    for (key, value) in G.items():
        if len(value) == value_max:
            return key

# Fonction qui crée un graphe à n sommet avec une probabilité d'apparition d'arêtes de p
def create_graph_random(n, p):
    l_aretes = []
    for i in range(n):
        for j in range(i, n):
            q = random()
            if q <= p: # arete presente
                l_aretes.append([str(i), str(j)])
    l_sommets = [str(i) for i in range(n)]
    return create_graph(n, l_sommets, len(l_aretes), l_aretes)

# Fonction qui compte le nombre d'arête dans un graphe
def compte_nb_aretes(G):
    cmp = 0
    for value in G.values():
        cmp += len(value)
    return cmp//2

# Algorithme couplage
def algo_couplage(G):
    C = set()
    #n = compte_nb_aretes(G)
    for key,value in G.items():
        for i in value:
            if key not in C and i not in C:
                C.add(key)
                C.add(i)
    return C

# Fonction qui renvoie l'algorithme couplage 
# où les sommets formant une arête sont sous la formes d'un tuple, rangées dans une liste
def couplage(C) :
    l = []
    C_2 = list(C)
    i=0
    while i < (len(C_2)-1) :
        # C_2[i] : sommet 
        # (C_2[i], C_2[i+1]) : couple de sommet formant une arête
        l.append((C_2[i], C_2[i+1]))
        i+=2
        
    return l

# Fonction qui renvoie la liste des arêtes de G
def list_aretes(G):
    aretes = []
    for key,value in G.items():
        for v in value:
            if [key,v] not in aretes and [v,key] not in aretes:
                aretes.append([key,v])
    return aretes

# Algorithme Glouton
def algo_glouton(G):
    G1 = copy.deepcopy(G)
    C = set()
    E = list_aretes(G1)
    while E != []:
        v = degre_max(G1)
        C.add(v)
        G1 = delete_sommet(G1, v)
        E = list_aretes(G1)
    return C


# pour calculer le temps
def calcul_temps_algo_glouton(g) :
    
    tps1 = time.time()
    algo_glouton(g)
    tps2 = time.time()
    return tps2 - tps1

def calcul_temps_algo_couplage(g) :
    
    tps1 = time.time()
    algo_couplage(g)
    tps2 = time.time()
    return tps2 - tps1

def calcul_temps_branchement(G, C) :
    
    tps1 = time.time()
    branchement(G, C)
    tps2 = time.time()
    return tps2 - tps1

     
# Branchement, question 4.1
def branchement(G, C) :
    L_A = list_aretes(G)
    # condition d'arrêt : liste d'arêtes vide
    if not L_A:
        return C
    
    e = L_A[0]  # e = {u, v}
    u, v = e[0], e[1]
    
    # Branche gauche : on met u dans C
    G1 = copy.deepcopy(G)
    C1 = C + [u]
    G1 = delete_sommet(G1, u)
    res1 = branchement(G1, C1)
    
    # Branche droite : on met v dans C
    G2 = copy.deepcopy(G)
    C2 = C + [v]
    G2 = delete_sommet(G2, v)
    res2 = branchement(G2, C2)
    
    return res1 if (len(res1) <= len(res2) and res1 != []) else res2

# Fonction qui retourne le maximum entre les bornes b1, b2, b3
def bornes(G, M, C) :
    # m = nombre d'arêtes de G
    m = len(list_aretes(G))
    delta = max(degre(G))
    # partie supérieur 
    b1 = math.ceil( m / int(delta) )
    
    b2 = len(M) / 2
    
    # n = nombre de sommets de G
    n = len([sommet for sommet,aretes in G.items()])
    b3 = (2*n - 1 - np.sqrt((2*n - 1)**2 - 8*m) ) / 2 # la partie dans la racine carré est >= 0, pas besoin de vérification
    
    return max([b1, b2, b3])

# Fonction qui montre la validité des bornes d'un graphe
def valide(b_max, C) :
    return len(C) >= b_max

def couverture_couplage(G):
    l = []
    M = list(algo_couplage(G))
    for i in range(0, len(M), 2):
        l.append(M[i])
    return l

# Branchement, question 4.2.2
def branchement2(G, C, borne_sup) :
    L_A = list_aretes(G)
    # condition d'arrêt : liste d'arêtes vide
    if not L_A:
        return C
    
    # Calcul d'une solution réalisable
    M = couplage(algo_couplage(G))
    # Calcul d'une borne inférieure
    borne_inf = bornes(G, M, C)
    # Solution non optimale
    #print("sup == inf :", borne_sup == borne_inf)
    print("borne sup :", borne_sup, "| borne inf :", borne_inf)
    if borne_sup <= borne_inf:
        print("on elague")
        return couverture_couplage(G)
    
    e = L_A[0]  # e = {u, v}
    u, v = e[0], e[1]
    
    # Branche gauche : on met u dans C
    G1 = copy.deepcopy(G)
    C1 = C + [u]
    G1 = delete_sommet(G1, u)
    res1 = branchement2(G1, C1, borne_sup)
    
    # Branche droite : on met v dans C
    G2 = copy.deepcopy(G)
    C2 = C + [v]
    G2 = delete_sommet(G2, v)
    res2 = branchement2(G2, C2, borne_sup)
    
    return res1 if (len(res1) <= len(res2) and res1 != []) else res2


def branchement_amelioration(G, C) :
    L_A = list_aretes(G)
    # condition d'arrêt : liste d'arêtes vide
    if not L_A:
        return C
    
    e = L_A[0]  # e = {u, v}
    u, v = e[0], e[1]
    
    # Branche gauche : on met u dans C
    G1 = copy.deepcopy(G)
    C1 = C + G1[v]
    G1 = delete_sommet(G1, u)
    G1 = delete_liste_sommets(G1, G1[v])
    print('branche gauche -----')
    print('u = ', u)
    print('C = ', C1)
    print('G = ', G1)
    print('\n')
    res1 = branchement_amelioration(G1, C1)
    
    # Branche droite : on met v dans C
    G2 = copy.deepcopy(G)

    C2 = C + G2[u]
    G2 = delete_sommet(G2, v)
    G2 = delete_liste_sommets(G1, G2[u])
    print('branche droit -----')
    print('u = ', v)
    print('C = ', C2)
    print('G = ', G2)
    print('\n')
    res2 = branchement_amelioration(G2, C2)
    
    
    return res1 if (len(res1) <= len(res2) and res1 != []) else res2

# #------------------Main------------------

nb_sommets, sommets, nb_aretes, aretes = read_file("exempleinstance.txt")
G = create_graph(nb_sommets, sommets, nb_aretes, aretes)
# # algo_glouton
# print("solution de algo glouton = ")
# C = algo_couplage(G)
# print(C)
# print("\n")

# print(couplage(C))

# #------------------Test les algo------------------

# # algo_glouton
# print("solution de algo glouton = ")
# print(algo_glouton(G))
# print("\n")
# # algo_couplage
# G = create_graph(nb_sommets, sommets, nb_aretes, aretes)
# print("solution de algo couplage = ")
# print(algo_couplage(G))
# print("\n")

# #------------------Test sur les temps de calculs------------------

graphe_alea = create_graph_random(1000, 0.0001)
somme=0
for s, a in graphe_alea.items():
    if a != []:
        somme+=1
print(somme)     
        
tmps1 = calcul_temps_algo_glouton(graphe_alea)
tmps2 = calcul_temps_algo_couplage(graphe_alea)
print('temps 1 = ', tmps1)
print('temps 2 = ', tmps2)

# #------------------Test sur les courbes------------------

# test de temps de calcul
# print(calcul_temps(5000, 0.5))
# tab_glouton = []
# tab_couplage = []
# tab_x = []
# i = 0
# Nmax = 100

# while(i < 10) :
#     tab_x.append(i* Nmax / 10)
#     g = create_graph_random(i* Nmax // 10, 0.5) 
#     tab_glouton.append( calcul_temps_algo_glouton(g))
#     tab_couplage.append( calcul_temps_algo_couplage(g))
#     i+=1
 

# plt.plot( [ np.log(i) for i in tab_glouton] , [ np.log(i) for i in tab_x] , label='glouton')
# plt.title('Tab_glouton')
# plt.show()

# #------------------Affichage des tests sur les temps de calculs------------------

# for i in range(1, 11):
#     x = i* Nmax // 10
#     g = create_graph_random(x, 0.5)
#     print("Nombre de sommets : ", x)
#     print("Temps de l'algo glouton : ", calcul_temps_algo_glouton(g))
#     print("Temps de l'algo couplage : ", calcul_temps_algo_couplage(g))

# #------------------Affichage des tests sur la qualite des solutions------------------

# for i in range(1, 11):
#     x = i* Nmax // 10
#     g = create_graph_random(x, 0.5)
#     print("Nombre de sommets : ", x)
#     print("Taille du résultat de l'algo glouton : ", len(algo_glouton(g)))
#     print("Taille du résultat de l'algo couplage : ", len(algo_couplage(g)))


# #------------------Test Branchement------------------

# # Exemple simple

# graphe_simple = create_graph(4, ['1', '2', '3', '4'], 3, [['1', '2'], ['2', '4'], ['2', '3']])
# print(branchement(graphe_simple, []))
# print("taille couplage : ", len(couplage(algo_couplage(graphe_simple))))
# print(branchement2(graphe_simple, [], len(couplage(algo_couplage(graphe_simple)))))
# print(branchement(G, C, S, 0))

# #------------------Test temps de calculs Branchement------------------

# graphe_alea = create_graph_random(18, 0.5)
# print("temps de calculs pour la fonction branchement : ")
# print(calcul_temps_branchement(graphe_alea, []))

# #------------------Courbe temps de calculs Branchement------------------
# i = 0
# tab = []
# Nmax = 15
# p = 1/np.sqrt(Nmax)
# Nmax_list = [n for n in range(Nmax)]  

# while (i < Nmax) :
   
#     g = create_graph_random(i * Nmax / 10, p)
#     tmps = calcul_temps_branchement(g, [])
#     tab.append(tmps)
#     i+=1
    

# plt.plot(Nmax_list, tab)
# plt.title('Temps de calculs pour la fonction branchement en fonction de Nmax')
# plt.xlabel('Nmax')
# plt.ylabel('Temps de calculs')
# plt.grid()
# plt.show()


# #------------------Validité des bornes------------------

# C = algo_glouton(G)
# M = couplage(algo_couplage(G))

# print(G, C , M)
# print("\n")
# b_max = bornes(G, M, C)
# print(valide(b_max, C))
# print(b_max)

# #------------------Test Branchement_amelioration------------------

# Exemple simple

# graphe_simple = create_graph(4, ['1', '2', '3', '4'], 3, [['1', '2'], ['2', '4'], ['2', '3']])
# print(branchement_amelioration(graphe_simple, []))
