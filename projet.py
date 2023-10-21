# -*- coding: utf-8 -*-
from __future__ import division
from random import *
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import copy
 

def read_file(file_path):
    """
    Lit le fichier et renvoie les informations concernant le graphe qui y sont contenues

    Parametres :
        * file_path : str - chemin vers le fichier à lire

    Valeurs de retour :
        * nb_sommets : int - nombre de sommets du graphe
        * sommets : list[str] - liste d'étiquettes des noms des sommets du graphe
        * nb_aretes : int - nombre d'arêtes du graphe
        * aretes : list[(str, str)] - liste de couples de sommets, sachant qu'un couple représente une arête
    """
    file = open(file_path, "r") # ouverture du fichier en mode lecture
    lignes = file.readlines() # liste qui contient toutes les lignes du fichier
    nb_sommets = 0
    sommets = []
    nb_aretes = 0
    aretes = []
    flag = 0 # flag qui permet d'indiquer la catégorie de données qui sont lues (1 : nb_sommets | 2 : sommets | 3 : nb_aretes | 4 : aretes)
    for ligne in lignes: # parcours des lignes
        # Changement de catégories
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
            # Interprétation et traitement des données lues en fonction de la catégorie indiquée par le flag
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
    """
    Crée et renvoie un graphe en fonction des caractéristiques données par les paramètres

    Paramètres :
        * nb_sommets : int - nombre de sommets du graphe
        * sommets : list[str] - liste d'étiquettes des noms des sommets du graphe
        * nb_aretes : int - nombre d'arêtes du graphe
        * aretes : list[(str, str)] - liste de couples de sommets, sachant qu'un couple représente une arête

    Valeur de retour :
        * G : dict(str : list[str]) - graphe, qui est un dictionnaire d'arêtes, ayant pour clé un sommet, et pour valeur une liste de ses voisins
    """
    G = dict()
    for sommet in sommets: # création des sommets
        G[str(sommet)] = []
    for arete in aretes: # ajout des arêtes
        G[arete[0]].append(arete[1])
        G[arete[1]].append(arete[0])
    return G

def delete_sommet(G, v):
    """
    Supprime le sommet v du graphe G, ainsi que toutes ses arêtes.
    Cette fonction modifie définitivement le graphe d'origine donné en paramètre.

    Paramètres :
        * G : dict(str : list[str]) - graphe à mettre à jour
        * v : str - sommet à supprimer
    
    Valeur de retour :
        * G : dict(str : list[str]) - graphe mis à jour
    """
    if v in G:
        del G[v] # suppression du sommet v
        for l_value in G.values():
            if v in l_value: # suppression des aretes qui ont pour extrémité v
                l_value.remove(v)
    return G

def delete_liste_sommets(G, lv):
    """
    Supprime les sommets dans lv, du graphe G, ainsi que toutes leurs arêtes.
    Cette fonction modifie définitivement le graphe d'origine donné en paramètre.

    Paramètres :
        * G : dict(str : list[str]) - graphe à mettre à jour
        * lv : list(str) - liste de sommets à supprimer
    
    Valeur de retour :
        * G : dict(str : list[str]) - graphe mis à jour
    """
    for v in lv:
        G = delete_sommet(G, v)
    return G

def degre(G):
    """
    Renvoie les degrés des sommets du graphe G

    Paramètres :
        * G : dict(str : list[str]) - graphe
    
    Valeur de retour :
        * _ : list(int) - liste des degrés des sommets du graphe
    """
    return [len(value) for value in G.values()]

def degre_max(G):
    """
    Renvoie le sommet qui a le degré maximum dans G (le premier en ordre d'apparition s'il y en a plusieurs)

    Paramètres : 
        * G : dict(str : list[str]) - graphe
    
    Valeur de retour :
        * key : str - le premier sommet de degré maximum dans G
    """
    value_max = max(degre(G)) # valeur du degré maximum
    for (key, value) in G.items():
        if len(value) == value_max:
            return key

def create_graph_random(n, p):
    """
    Renvoie un grpahe aléatoire de n sommets, avec une probabilité p d'avoir une arête entre deux sommets

    Paramètres:
        * n int - nombre de sommets
        * p float entre 0 et 1 - probabilité d'avoir une arête entre deux sommets

    Valeur de retour:
        * G dict(str : list[str]) - graphe aléatoire
    """
    l_aretes = []
    for i in range(n):
        for j in range(i, n):
            q = random()
            if q <= p: # arete presente
                l_aretes.append([str(i), str(j)])
    l_sommets = [str(i) for i in range(n)]
    return create_graph(n, l_sommets, len(l_aretes), l_aretes)

def compte_nb_aretes(G):
    """
    Renvoie le nombre d'arêtes du graphe G
    
    Paramètres:
        * G dict(str : list[str]) - graphe
    
    Valeur de retour:
        * cmp int - nombre d'arêtes du graphe G
    """
    cmp = 0
    for value in G.values():
        cmp += len(value)
    return cmp//2

def algo_couplage(G):
    """
    Renvoie un couplage de G sous forme d'un ensemble de sommets

    Paramètres:
       * G dict(str : list[str]) - graphe

    Valeur de retour:
        * C set(str) - couplage de G
    """
    C = set()
    for key,value in G.items():
        for i in value:
            if key not in C and i not in C:
                C.add(key)
                C.add(i)
    return C

def couplage(C) :
    """
    Renvoie un couplage sous forme d'une liste de tuples de sommet ( = arêtes)

    Paramètres:
       * C set(str) - couplage de G

    Valeur de retour:
        * l list(tuple(str, str)) - couplage de G sous forme de liste de tuples
    """
    l = []
    C_2 = list(C)
    i=0
    while i < (len(C_2)-1) :
        l.append((C_2[i], C_2[i+1]))
        i+=2
        
    return  l

def list_aretes(G):
    """
    Renvoie la liste des arêtes du graphe G
    
    Paramètres:
        * G dict(str : list[str]) - graphe
        
    Valeur de retour:
        * aretes list(tuple(str, str)) - liste des arêtes du graphe G
    """
    aretes = []
    for key,value in G.items():
        for v in value:
            if [key,v] not in aretes and [v,key] not in aretes:
                aretes.append([key,v])
    return aretes

def algo_glouton(G):
    """
    Renvoie une couverture de G sous forme d'un ensemble de sommets

    Paramètres:
         * G dict(str : list[str]) - graphe

    Valeur de retour:
        * C set(str) - couverture de G
    """
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
    """
    Calcul le temps d'execution de l'algo glouton

    Paramètres:
        * g : dict(str : list[str]) - graphe

    Valeur de retour:
        * tps2 - tps1 : float - temps d'execution de l'algo glouton
    """
    tps1 = time.time()
    algo_glouton(g)
    tps2 = time.time()
    return tps2 - tps1

def calcul_temps_algo_couplage(g) :
    """
    Calcul le temps d'execution de l'algo couplage

    Paramètres:
        * g : dict(str : list[str]) - graphe

    Valeur de retour:
        * tps2 - tps1 : float - temps d'execution de l'algo couplage
    """
    tps1 = time.time()
    algo_couplage(g)
    tps2 = time.time()
    return tps2 - tps1

def calcul_temps_branchement(G, C) :
    """
    Calcul le temps d'execution de branchement

    Paramètres:
        * G : dict(str : list[str]) - graphe
        * C : list(str) - couverture partielle

    Valeur de retour:
        * tps2 - tps1 : float - temps d'execution de branchement
    """
    tps1 = time.time()
    branchement(G, C)
    tps2 = time.time()
    return tps2 - tps1

     

def branchement(G, C) :
    """
    Renvoie une couverture minimale d'un graphe G en parcourant toutes les possibilités

    Paramètres:
        * G (dict(str : list[str])): graphe
        * C (list(str)): couverture partielle

    Valeur de retour:
        * res1 (list(str)): couverture minimale
    """
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

def bornes(G, M, C) :
    """
    Renvoie les bornes d'un graphe G

    Paramètres:
        * G (dict(str : list[str])): graphe
        * M (list(tuple(str, str))): couplage
        * C (list(str)): couverture partielle

    Valeur de retour:
        * max([b1, b2, b3]) (int): la borne supérieure
    """
    # m = nombre d'arêtes de G
    m = len(list_aretes(G))
    delta = max(degre(G))
    # partie supérieur 
    if delta == 0: # pour éviter une division par zero dans le cas où le graphe est vide
        b1 = 0
    else:
        b1 = math.ceil( m / int(delta) )
    
    b2 = len(M) / 2
    
    # n = nombre de sommets de G
    n = len(G)
    if (2*n - 1)**2 - 8*m >= 0:
        b3 = (2*n - 1 - np.sqrt((2*n - 1)**2 - 8*m) ) / 2
    else:
        b3 = 0
    return max([b1, b2, b3])


def valide(b_max, C) :
    """
    Renvoie True si les bornes sont valides

    Paramètres:
        * b_max (int): la borne supérieure
        * C (list(str)): couverture partielle

    Valeur de retour:
        * True si les bornes sont valides, False sinon
    """
    return len(C) >= b_max

def couverture_couplage(G):
    """
    Renvoie une couverture de G sous forme d'un ensemble de sommets

    Paramètres:
        * G (dict(str : list[str])): graphe

    Valeur de retour:
        * C (set(str)): couverture de G
    """
    l = []
    M = list(algo_couplage(G))
    for i in range(0, len(M), 2):
        l.append(M[i])
    return l

def branchement2(G, C) :
    """
    Renvoie une couverture minimale d'un graphe G en parcourant toutes les possibilités où en chaque noeuds, 
    - on calcul une couverture réalisable avec l'algorithme couplage
    - on calcul une borne inférieure avec la fonction bornes

    Paramètres:
        * G (dict(str : list[str])): graphe
        * C (list(str)): couverture partielle

    Valeur de retour:
        * res1 (list(str)): couverture minimale
    """
    # borne_sup = variable globale
    global borne_sup
    L_A = list_aretes(G)
    # condition d'arrêt : liste d'arêtes vide
    if not L_A:
        # Mise à jour de la borne supérieure avec une meilleure solution réalisable
        if borne_sup > len(C):
            borne_sup = len(C)
        return C
    
    # Calcul d'une solution réalisable
    M = couplage(algo_couplage(G))
    # Calcul d'une borne inférieure
    borne_inf = bornes(G, M, C)
    # Solution non optimale
    if borne_sup <= borne_inf:
        return [] # on élague

    e = L_A[0]  # e = {u, v}
    u, v = e[0], e[1]
    
    # Branche gauche : on met u dans C
    G1 = copy.deepcopy(G)
    C1 = C + [u]
    G1 = delete_sommet(G1, u)
    M1 = algo_couplage(G)
    borne_inf_1 = bornes(G, M1, C1)
    res1 = branchement2(G1, C1)
    
    # Branche droite : on met v dans C
    G2 = copy.deepcopy(G)
    C2 = C + [v]
    G2 = delete_sommet(G2, v)
    M2 = algo_couplage(G)
    borne_inf_2 = bornes(G, M2, C2)
    res2 = branchement2(G2, C2)
   
    return res1 if (len(res1) <= len(res2) and res1 != []) else res2


def branchement_amelioration(G, C) :
    """
    Reprend l'algorithme de branchemenent2 et renvoie une couverture minimale d'un graphe G en parcourant toutes les possibilités où,
    - on ajoute tout les voisins de u dans la couverture de la branche droite
    
    Paramètres:
        * G (dict(str : list[str])): graphe
        * C (list(str)): couverture partielle

    Valeur de retour:
        * res1 (list(str)): couverture minimale
    """
    # borne_sup = variable globale
    global borne_sup
    borne_sup = float("inf")  # Initialisation à l'infini positif
    
    L_A = list_aretes(G)
    # condition d'arrêt : liste d'arêtes vide
    if not L_A:
        # Mise à jour de la borne supérieure avec une meilleure solution réalisable
        if borne_sup > len(C):
            borne_sup = len(C)
        return C
    
    # Calcul d'une solution réalisable
    M = couplage(algo_couplage(G))
    # Calcul d'une borne inférieure
    borne_inf = bornes(G, M, C)
    # Solution non optimale
    if borne_sup <= borne_inf:
        return [] # on élague

    e = L_A[0]  # e = {u, v}
    u, v = e[0], e[1]

    # Vérifier si les clés existent avant de les supprimer
    if u in G:
        # Branche gauche : on met u dans C
        G1 = copy.deepcopy(G)
        C1 = C + [u]
        G1 = delete_sommet(G1, u)
        res1 = branchement_amelioration(G1, C1)
    
    if v in G:
        # Branche droite : on met v dans C, et tous les voisins de u
        G2 = copy.deepcopy(G)
        C2 = C + G2[u] # on ajoute les voisins de u (v compris) dans C
        G2 = delete_sommet(G2, v)
        G2 = delete_liste_sommets(G2, [u] + G2[u]) # on supprime u et ses voisins de G2
        res2 = branchement_amelioration(G2, C2)
   
    return res1 if (len(res1) <= len(res2) and res1 != []) else res2


def branchement_amelioration2(G, C) :
    """
    Reprend l'algorithme de branchement_amelioration et :
    - on choisi u de manière à ce que ce soit le sommet de degré maximum
    
    Paramètres:
        * G (dict(str : list[str])): graphe
        * C (list(str)): couverture partielle
    
    Valeur de retour:
        * res1 (list(str)): couverture minimale
        
    """
    global borne_sup
    borne_sup = float("inf")  # Initialisation à l'infini positif

    L_A = list_aretes(G)
    # condition d'arrêt : liste d'arêtes vide
    if not L_A:
        # Mise à jour de la borne supérieure avec une meilleure solution réalisable
        if borne_sup > len(C):
            borne_sup = len(C)
        return C
    
    # Calcul d'une solution réalisable
    M = couplage(algo_couplage(G))
    # Calcul d'une borne inférieure
    borne_inf = bornes(G, M, C)
    # Solution non optimale
    if borne_sup <= borne_inf:
        return [] # on élague

    u = degre_max(G)  # u = sommet de degre max
    v = G[u][0]  # v = voisin de u

    # Vérifier si les clés existent avant de les supprimer
    if u in G:
        # Branche gauche : on met u dans C
        G1 = copy.deepcopy(G)
        C1 = C + [u]
        G1 = delete_sommet(G1, u)
        res1 = branchement_amelioration2(G1, C1)

    if v in G:
        # Branche droite : on met v dans C, et tous les voisins de u
        G2 = copy.deepcopy(G)
        C2 = C + G2[u] if u in G else C
        if u in G:
            G2 = delete_sommet(G2, v)
        if u in G and u in G2:
            G2 = delete_liste_sommets(G2, [u] + G2[u])  # on supprime u et ses voisins de G2
        res2 = branchement_amelioration2(G2, C2)

    return res1 if (len(res1) <= len(res2) and res1 != []) else res2


def rapport_approx_glouton(n, p) :
    """
    Renvoie le rapport d'approximation de l'algorithme glouton
    
    Paramètres:
        * n (int): nombre de sommets
        * p (float): probabilité d'avoir une arête entre deux sommets
        
    Valeur de retour:
        * len(couverture_glouton) / len(couverture_op) (float): rapport d'approximation 
        
    """
    G = create_graph_random(n, p)
    borne_sup = len(G)
    # calcul de la couverture de l'algo glouton
    couverture_glouton = algo_glouton(G)
    # calcul de la couverture optimal
    couverture_op = branchement_amelioration2(G, [])

    # calcul du rapport d'approximation 
    if len(couverture_op) == 0:
        return 0
    return len(couverture_glouton) / len(couverture_op)

def rapport_approx_couplage(n, p) :
    """
    Renvoie le rapport d'approximation de l'algorithme couplage
    
    Paramètres:
        * n (int): nombre de sommets
        * p (float): probabilité d'avoir une arête entre deux sommets
        
    Valeur de retour:
        * len(couverture_c) / len(couverture_op) (float): rapport d'approximation 
        
    """
    G = create_graph_random(n, p)
    borne_sup = len(G)
    # calcul de la couverture de l'algo couplage
    couverture_c = couplage(couverture_couplage(G))
    # calcul de la couverture optimal
    couverture_op = branchement_amelioration2(G, [])

    # calcul du rapport d'approximation
    if len(couverture_op) == 0:
        return 0 
    return len(couverture_c) / len(couverture_op)
    



# #------------------Main------------------

# nb_sommets, sommets, nb_aretes, aretes = read_file("exempleinstance.txt")
# G = create_graph(nb_sommets, sommets, nb_aretes, aretes)
# print(G)
# print("degre :", degre(G))
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
#print("taille couplage : ", len(couplage(algo_couplage(graphe_simple))))
# borne_sup = len(graphe_simple)
# print(branchement2(graphe_simple, []))
# print(branchement(G, C, S, 0))
# print(branchement_amelioration(graphe_simple, []))
# print(branchement_amelioration(G, []))
# print(branchement_amelioration2(graphe_simple, []))
# print(branchement_amelioration2(G, []))

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



# #------------------Qualité des agorithmes------------------

# # Contre-Exemple qui montre que algo glouton est pas optimal
# sommets = [str(i) for i in range(1, 15)]
# aretes = [ ['1', '7'], 
#            ['1', '10'], 
#            ['1', '12'], 
#            ['1', '13'], 
#            ['1', '14'], 
#            ['2', '10'], 
#            ['2', '12'], 
#            ['2', '13'], 
#            ['2', '14'],
#            ['3', '8'], 
#            ['3', '10'], 
#            ['3', '12'], 
#            ['3', '13'], 
#            ['3', '14'],
#            ['4', '8'], 
#            ['4', '11'], 
#            ['4', '12'], 
#            ['4', '13'], 
#            ['4', '14'],
#            ['5', '11'], 
#            ['5', '9'],  
#            ['5', '13'], 
#            ['5', '14'],
#            ['6', '11'], 
#            ['6', '9'],  
#            ['6', '14'],
           
#           ]

# G = create_graph(14, sommets, len(aretes), aretes)
# # calcul de la couverture de l'algo glouton
# couverture_glouton = algo_glouton(G)
# print("couverture de l'algorithme glouton : ", couverture_glouton)
# # calcul de la couverture optimal
# couverture_op = branchement_amelioration2(G, [])
# print("couverture de l'algorithme branchement : ", couverture_op)
# print("\n")
# print("rapport d'approximation", len(couverture_glouton) / len(couverture_op))

# #-----------------------------------------------------------------
# tab_glouton = []
# p = 0.5
# n = 60
# for i in range(1, n) :
#     tab_glouton.append(rapport_approx_glouton(i, p))

# print(tab_glouton)
# print("pire rapport d'approximation pour l'algo glouton : ", max(tab_glouton))

# plt.plot([i for i in range(1, n)], tab_glouton, label='glouton')
# plt.title('Rapport d approximation en fonction de n')
# plt.xlabel('n')
# plt.ylabel('Rapport d approximation')
# plt.grid()
# plt.show()

# tab_couplage = []
# p = 0.4
# n = 20
# for i in range(1, n) :
#     tab_couplage.append(rapport_approx_couplage(i, p))


# print("pire rapport d'approximation pour l'algo couplage : ", max(tab_couplage))

# plt.plot([i for i in range(1, n)], tab_couplage, label='couplage')
# plt.title('Rapport d approximation en fonction de n')
# plt.xlabel('n')
# plt.ylabel('Rapport d approximation')
# plt.grid()
# plt.show()

# print("rapport d'approximation pour l'algo glouton : ", rapport_approx_glouton(70, 0.5))
#print("rapport d'approximation pour l'algo couplage : ", rapport_approx_couplage(50, 0.5))
    
