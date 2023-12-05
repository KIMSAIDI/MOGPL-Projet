# -*- coding: utf-8 -*-
import sys
import random
import copy
import matplotlib.pyplot as plt
from exemples import *


# Question 1
def Bellman_Ford(G, s, ordre):
    """
    Renvoie l'algorithme de Bellman_Ford.
    Le graphe ne contient aucun circuit négatif (d'après énoncé).

    Paramètres:
        G : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire. 
            Les keys (int) sont tous les sommets du graphes. 
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont representés sous la formes d'un tuples avec comme deuxième element le poid associé
        
        s : int - sommet

        ordre : list[int] - un ordre de sommets à parcourir

    Valeurs de retour
        dict(int : list[(int, int)])
            Un dictionnaire représentant l'arborescence des plus courts chemins (chemins considérés par l'algorithme à l'itération finale)
            Le format est celui d'un graphe classique (voir l'entrée), les arêtes ne sont pas valuées (=0) mais le poids est gardé pour avoir le même format

        int
            Le nombre d'itérations effectuées par l'algorithme

    """
    # n : nombre de sommet dans G
    n = len(G)
    # nb_it : nombre d'itérations nécessaire avant la convergence de l'algorithme
    nb_it = 0
    # d : liste représentant les longueurs des chemins entre s et les autres sommets
    # on initialise pour tous les sommets dans G, la longueur = infini et pour s = 0
    d = {u: 0 if u == s else sys.maxsize for u in G.keys()}

    parents = {u: -1 for u in G.keys()}  # Parents lie chaque sommet au parent duquel on vient (dans le plus court chemin) -1 = pas de parent

    # boucle principale
    for i in range(n - 1):
        # boolean qui va nous permettre de savoir si l'algorithme a convergé
        boolean = True
        # u : sommet
        # v : liste des arcs sortant de u avec le poid correspondant
        for u in ordre:
            v = G[u]
            # arcs : tuple(int, int) ; arcs[0] : sommet, arcs[1] : poids
            for arcs in v:
                if d[u] + arcs[1] < d[arcs[0]]:
                    # si on rentre dans cette condition : l'algorithme n'a toujours pas convergé
                    boolean = False
                    # on update la valeur de la longueur
                    d[arcs[0]] = d[u] + arcs[1]

                    # On update le parent
                    parents[arcs[0]] = u

        if boolean:  # l'algorithme a convergé
            break
        else:
            nb_it += 1

    # On reconstruit l'arborescence des chemins les plus courts à partir des parents
    arborescence = {u: [] for u in G.keys()}

    for u, parent in parents.items():
        if parent != -1:
            arborescence[parent].append((u, 0))

    return arborescence, nb_it


# Question 2
def nombre_entrants(G):
    """
    Calcule le nombre d'arcs entrants pour chaque sommet du graphe

    Paramètre :
        G : dict(int : list[(int, int)])
        Graphe, voir GloutonFas pour les détails de représentation

    Valeur de retour
        dict(int : int)
        Un dictionnaire liant chaque sommet du graphe à son nombre d'arcs entrants
    """
    compteur = {k: 0 for k in G.keys()}

    # On incrémente le sommet de destination de chaque arc du graphe
    for arcs in G.values():
        for arc in arcs:
            if arc[0] in G:  # On vérifie si le sommet de destination n'a pas été supprimé
                compteur[arc[0]] += 1

    return compteur


def sources(G):
    """
    Cherche une source dans un graphe G

    Paramètre :
        G : dict(int : list[(int, int)])
        Graphe, voir GloutonFas pour les détails de représentation

    Valeur de retour
        List([int])
        Une liste comportant les sources du graphe
    """
    entrants = nombre_entrants(G)

    liste_sources = []

    # On cherche un sommet avec aucun arc entrant (= source)
    for key, val in entrants.items():
        if val == 0:
            liste_sources.append(key)

    return liste_sources


def puits(G):
    """
    Cherche un puits dans un graphe G

    Paramètre :
        G : dict(int : list[(int, int)])
        Graphe, voir GloutonFas pour les détails de représentation

    Valeur de retour
        Une liste comportant les sources du graphe
    """
    liste_puits = []

    # On cherche un sommet avec aucun arc sortant (= puits)
    for key, val in G.items():
        if not val:
            liste_puits.append(key)

    return liste_puits


def max_delta(G):
    """
    Calcule la fonction max δ(u) pour chaque sommet u

    Paramètre :
        G : dict(int : list[(int, int)])
        Graphe, voir GloutonFas pour les détails de représentation

    Valeur de retour
        int | None
        Le numéro d'un sommet u tel que δ(u) soit maximal, None si le graphe ne comporte pas de sommet
    """
    if not G:  # Si le graphe est vide il n'y a pas de sommet à retourner
        return

    entrants = nombre_entrants(G)
    sortants = {key: len(val) for key, val in G.items()}

    delta = {key: sortants[key] - entrants[key] for key in G.keys()}

    return max(delta, key=lambda x: delta[x])


def GloutonFas(G):
    """
    Implémentation de l'algorithme GloutonFas

    Paramètre :
        G : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire.
            Les keys (int) sont tous les sommets du graphe.
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé

    Valeur de retour :
        list([int])
        Permutation des sommets du graphe, le ième élément de la liste représente le ième sommet de la permutation
    """
    G = G.copy()  # On ne souhaite pas modifier le graphe originel
    s1, s2 = [], []

    while G:  # Tant que le graphe n'est pas vide
        u = sources(G)  # U est la liste de tous les sommets sources de G
        while u:  # Tant qu'on trouve des sources
            for sommet in u:
                s1.append(sommet)
                del G[sommet]
            u = sources(G)

        u = puits(G)  # U est la liste de tous les sommets puits de G
        while u:  # Tant qu'on trouve des puits
            for sommet in u:
                s2.append(sommet)  # Ajouter un élément en début de liste est très coûteux, on les ajoute donc à la fin et on inversera s2
                del G[sommet]
            u = puits(G)

        u = max_delta(G)  # u est le sommet ayant la différence entre son degré sortant et son degré entrant maximale
        if u:  # Si un tel noeud existe (= le graphe n'est pas vide)
            s1.append(u)
            del G[u]

    return s1 + s2[::-1]  # On inverse s2 (on l'a construite à l'envers)


def creation_graphes(G) :
    """
    A partir de G, on crée 4 graphes : G1, G2, G3, H qui nous permet de tester l'efficacité des l'algorithmes 
    On choisit de manière uniforme et aléatoire les poids des arcs dans l'intervalle [-10, 10]

    Paramètres:
        G : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire.
            Les keys (int) sont tous les sommets du graphe.
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé
    
    Valeur de retour :
        G1, G2, G3, H : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire.
            Les keys (int) sont tous les sommets du graphe.
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé
    """

    G1 = copy.deepcopy(G)
    G2 = copy.deepcopy(G)
    G3 = copy.deepcopy(G)
    H = copy.deepcopy(G)

    for u, v in G1.items():
        for i in range(len(v)): # i : indice de l'arc
            v[i] = (v[i][0], random.randint(-10, 10))

    for u, v in G2.items():
        for i in range(len(v)):
            v[i] = (v[i][0], random.randint(-10, 10))

    for u, v in G3.items():
        for i in range(len(v)):
            v[i] = (v[i][0], random.randint(-10, 10))

    for u, v in H.items():
        for i in range(len(v)):
            v[i] = (v[i][0], random.randint(-10, 10))

    return G1, G2, G3, H


def union(G1, G2) :
    """
    Retourne l'union de deux graphes en choissisant les plus courts chemins
    On part du principe que G1 et G2 sont des graphes sans circuit négatif qui sont identiques (diffère seulement sur la valeur des poids des arcs)
    
    Paramètres:
        G1 dict(int : list[(int, int)]) - Graphe
        G2 dict(int : list[(int, int)]) - Graphe
    
    Valeur de retour :
        dict(int : list[(int, int)]) -
        union des deux graphes
    """
    G = {}  # Graphe de retour

    for sommet in G1.keys():
        G[sommet] = list(set(G1[sommet] + G2[sommet]))  # On concatène les listes d'arêtes des deux graphes (on caste en set au milieu pour enlever les doublons)
        
    return G

def ordre_tot(G, s):
    """
    Retourne un ordre <tot à partir d'un graphe

    Paramètre:
        G : dict(int : list[(int, int)]) - Le graphe
        s: int - la source depuis laquelle exécuter bellman_ford

    Valeur de retour :
        list[int] - L' ordre <tot déterminé
    """

    G1, G2, G3, H = creation_graphes(G)

    d1, nb_it1 = Bellman_Ford(G1, s, G.keys())
    d2, nb_it2 = Bellman_Ford(G2, s, G.keys())
    d3, nb_it3 = Bellman_Ford(G3, s, G.keys())

    print(f"{G1=}\n{d1=}, {nb_it1=}\n")
    print(f"{G2=}\n{d2=}, {nb_it2=}\n")
    print(f"{G3=}\n{d3=}, {nb_it3=}\n")

    T = union(d1, d2)
    T = union(T, d3)

    print(f"{T=}")

    return GloutonFas(T)

            
def create_graph_random(n, p) :
    """
    Renvoie un graphe aléatoire de n sommets, avec une probabilité p d'avoir une arête entre deux sommets
    """
    G = dict()
    
    # sommet source
    nb_arcs = random.randint(n//2, n)
    sommet_source = 0
    l_aretes = []
    for j in range(nb_arcs) :
        poids = random.randint(-10, 10)
        successeur = random.randint(0, n-1)
       
        while successeur == sommet_source :
            successeur = random.randint(0, n-1)
        
        l_aretes.append((successeur, poids))
     
    G[sommet_source] = l_aretes
    
   
    for i in range(1, n) :
        l_aretes = []
        nb_arcs = random.randint(0, n)
        for j in range(nb_arcs) :
            # test si arc présentes
            q = random.random()
            if q <= p :
                poids = random.randint(-10, 10)
                successeur = random.randint(1, n)
                # si jamais le successeur = au sommet courant
                # pour éviter les boucles
                while successeur == i :
                    successeur = random.randint(1, n)
                l_aretes.append((successeur, poids))
        G[i] = l_aretes
    
    
    return G, sommet_source

def detection_circuit_negatif(G, s) :
    """
    Detecte grâce a l'algorithme Bellman-Ford si G est un circuit négatif
    """
    n = len(G)
    nb_it = 0
    d = {u: 0 if u == s else sys.maxsize for u in G.keys()}

    # boucle principale
    for i in range(n):
        boolean = True
        # u : sommet
        # v : liste des arcs sortant de u avec le poid correspondant
        for u in ordre:
            v = G[u]
            # arcs : tuple(int, int) ; arcs[0] : sommet, arcs[1] : poids
            for arcs in v:
                if d[u] + arcs[1] < d[arcs[0]]:
                    # si on rentre dans cette condition : l'algorithme n'a toujours pas convergé
                    boolean = False
                    # on update la valeur de la longueur
                    d[arcs[0]] = d[u] + arcs[1]

        if boolean:  # l'algorithme a convergé
            break
        else:
            if nb_it == n :
                return True
            nb_it += 1
    
    return False

if __name__ == "__main__":
    G = exemple1()

    print(f"{G=}")

    # Sommet source
    s = 1

    arborescence, nb_it = Bellman_Ford(G, s, G.keys())
    print("arborescence = ", arborescence)
    print("Nombre iterations = ", nb_it)

    s = GloutonFas(G)
    print(f"{s=}")
    
    G = exemple2()
    print(f"\n -- Exemple 3 --\n\n{G=}\n")
    G1, G2, G3, H = creation_graphes(G)
    
    tot = ordre_tot(G, 0)
    print(f"{tot=}\n")

    # Questions 6 / 7
    s = 4

    arb_tot, nb_it_tot = Bellman_Ford(G, s, tot)
    ordre_rand = list(G.keys())
    random.shuffle(ordre_rand)
    arb_rand, nb_it_rand = Bellman_Ford(G, s, ordre_rand)

    print(f"{arb_tot=}\n{nb_it_tot=}\n")
    print(f"{arb_rand=}\n{nb_it_rand=}\n")
    
    
    # Question 9
    
    Nmax = 10
    p = 0.5
    tab = []
    for i in range(Nmax):
        n = i* Nmax // 10
        g = create_graph_random(n, p)
        tab.append(g)
        
    
    
