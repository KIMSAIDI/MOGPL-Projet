# -*- coding: utf-8 -*-

from random import *
import time
import matplotlib.pyplot as plt
import numpy as np
import copy

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

def create_graph(nb_sommets, sommets, nb_aretes, aretes):
    G = dict()
    for sommet in sommets:
        G[str(sommet)] = []
    for arete in aretes:
        G[arete[0]].append(arete[1])
        G[arete[1]].append(arete[0])
    return G

def delete_sommet(G, v):
    if v in G:
        del G[v]
        for l_value in G.values():
            if v in l_value:
                l_value.remove(v)
    return G

def delete_liste_sommets(G, lv):
    for v in lv:
        G = delete_sommet(G, v)
    return G

def degre(G):
    return [len(value) for value in G.values()]

def degre_max(G):
    value_max = max(degre(G))
    for (key, value) in G.items():
        if len(value) == value_max:
            return key

def create_graph_random(n, p):
    l_aretes = []
    for i in range(n):
        for j in range(i, n):
            q = random()
            if q <= p: # arete presente
                l_aretes.append([str(i), str(j)])
    l_sommets = [str(i) for i in range(n)]
    return create_graph(n, l_sommets, len(l_aretes), l_aretes)

def compte_nb_aretes(G):
    cmp = 0
    for value in G.values():
        cmp += len(value)
    return cmp//2

def algo_couplage(G):
    C = set()
    n = compte_nb_aretes(G)
    for key,value in G.items():
        for i in value:
            if key not in C and i not in C:
                C.add(key)
                C.add(i)
    return C

def list_aretes(G):
    aretes = []
    for key,value in G.items():
        for v in value:
            if [key,v] not in aretes and [v,key] not in aretes:
                aretes.append([key,v])
    return aretes

def algo_glouton(G):
    C = set()
    E = list_aretes(G)
    while E != []:
        v = degre_max(G)
        C.add(v)
        G = delete_sommet(G, v)
        E = list_aretes(G)
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


# def delete_sommet(G, v):
#     G2 = copy.deepcopy(G) # deepcopy pour ne pas modifier l'original (clé+liste de valeur)
#     del G2[v]
#     for l_value in G2.values():
#         for u in l_value:
#             if u == v:
#                 l_value.remove(u)
#     return G2


i = 1
def branchement(G, C, S, var) :
    global i  
    print('appel de la fonction numero :', i)
    i+=1
    print(' on est dans la fonction ', var)
    print('S = ', S)
    print(' C = ', C)
    print('\n')
    # un graphe G : dict()
    # une couverture C (la solution partielle ou complète) : list()
    # un ensemble de sommets à traiter S : list()
    # Cas d'arret
    if S == []: # feuille de l'arbre (tous les sommets sont traités)
        if list_aretes(G) == []: # graphe sans aretes donc la couverture est une solution réalisable
            return C 
        return [] # sinon solution non réalisable (renvoie une couverture vide)
    # Recursivite
    u = S[0] # sommet à traiter
    G2 = delete_sommet(G, u) # graphe sans le sommet u et toutes ses aretes
    
    res1 = branchement(G2, C+[u], S[1:], 1) # on prend u
    # print('s = ', S)
    # print('res1 = ', res1)

    res2 = branchement(G, C, S[1:], 2) # on ne prend pas u
   
    # Optimal : la plus petite couverture
    if res1 == []:
        return res2 
    elif res2 == []:
        return res1
    else:
        if len(res1) <= len(res2):
            return res1 
        return res2
    
     

def branchement2(G, C) :
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
    print('branche gauche -----')
    print('u = ', u)
    print('C = ', C1)
    print('\n')
    res1 = branchement2(G1, C1)
    
    # Branche droite : on met v dans C
    G2 = copy.deepcopy(G)
    C2 = C + [v]
    G2 = delete_sommet(G2, v)
    print('branche droit -----')
    print('u = ', v)
    print('C = ', C2)
    print('\n')
    res2 = branchement2(G2, C2)
    
    return res1 if len(res1) <= len(res2) else res2


#------------------Main------------------

nb_sommets, sommets, nb_aretes, aretes = read_file("exempleinstance.txt")
G = create_graph(nb_sommets, sommets, nb_aretes, aretes)

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

# graphe_alea = create_graph_random(100, 0.5)
# tmps1 = calcul_temps_algo_glouton(graphe_alea)
# tmps2 = calcul_temps_algo_couplage(graphe_alea)
# print('temps 1 = ', tmps1)
# print('temps 2 = ', tmps2)

# #------------------Test sur les courbes------------------

# test de temps de calcul
# print(calcul_temps(5000, 0.5))
# tab_glouton = []
# tab_couplage = []
# tab_x = []
# i = 0
# Nmax = 1000

# while(i < 10) :
#     tab_x.append(i* Nmax / 10)
#     g = create_graph_random(i* Nmax / 10, 0.5) 
#     tab_glouton.append( calcul_temps_algo_glouton(g))
#     tab_couplage.append( calcul_temps_algo_couplage(g))
#     i+=1
 

# plt.plot( [ np.log(i) for i in tab_glouton] , [ np.log(i) for i in tab_x] , label='glouton')
# plt.title('Tab_glouton')
# plt.show()


# plt.plot(tab_x, tab_couplage)
# plt.show()


# #------------------Test Branchement------------------

# print(branchement2(G, []))
# print("result = ")
# print(branchement(G, C, S, 0))


