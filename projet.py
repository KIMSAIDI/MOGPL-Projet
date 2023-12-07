# -*- coding: utf-8 -*-
import functools
import math
import sys
import random
import copy

from matplotlib import pyplot as plt

# import matplotlib.pyplot as plt
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


# def creation_graphes(G) :
#     """
#     A partir de G, on crée 4 graphes : G1, G2, G3, H qui nous permet de tester l'efficacité des l'algorithmes 
#     On choisit de manière uniforme et aléatoire les poids des arcs dans l'intervalle [-10, 10]

#     Paramètres:
#         G : dict(int : list[(int, int)]) -
#             Graphe, représenté sous la forme d'un dictionnaire.
#             Les keys (int) sont tous les sommets du graphe.
#             Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé
    
#     Valeur de retour :
#         G1, G2, G3, H : dict(int : list[(int, int)]) -
#             Graphe, représenté sous la forme d'un dictionnaire.
#             Les keys (int) sont tous les sommets du graphe.
#             Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé
#     """

#     G1 = copy.deepcopy(G)
#     G2 = copy.deepcopy(G)
#     G3 = copy.deepcopy(G)
#     H = copy.deepcopy(G)

#     for u, v in G1.items():
#         for i in range(len(v)): # i : indice de l'arc
#             v[i] = (v[i][0], random.randint(-10, 10))

#     for u, v in G2.items():
#         for i in range(len(v)):
#             v[i] = (v[i][0], random.randint(-10, 10))

#     for u, v in G3.items():
#         for i in range(len(v)):
#             v[i] = (v[i][0], random.randint(-10, 10))

#     for u, v in H.items():
#         for i in range(len(v)):
#             v[i] = (v[i][0], random.randint(-10, 10))

#     return G1, G2, G3, H

# fonction creation_graphes mais avec n graphes

def creation_graphes(G, n, s) :
    """
    A partir de G, on crée n graphes : G1, G2, ..., G(n-1), H qui nous permet de tester l'efficacité des algorithmes
    On choisit de manière uniforme et aléatoire les poids des arcs dans l'intervalle [-10, 10]

    Paramètres:
        G : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire.
            Les keys (int) sont tous les sommets du graphe.
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé
        n : le nombre de graphes aléatoires à créer
        s : la source du graphe

    Valeur de retour :
        graphes : list[dict(int : list[(int, int)])]
            Une liste de graphes G1, G2, ..., G(n-1) générés
        H : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire.
            Les keys (int) sont tous les sommets du graphe.
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont représentées sous la forme d'un tuple avec comme deuxième element le poids associé
    """

    graphes = [copy.deepcopy(G) for _ in range(n-1)]
    H = copy.deepcopy(G)

    for graphe in graphes:
        for u, v in graphe.items():
            for i in range(len(v)): # i : indice de l'arc
                v[i] = (v[i][0], random.randint(-5, 15))
        while detection_circuit_negatif(graphe, s):
            for u, v in graphe.items():
                for i in range(len(v)):  # i : indice de l'arc
                    v[i] = (v[i][0], random.randint(-5, 15))

    for u, v in H.items():
        for i in range(len(v)):
            v[i] = (v[i][0], random.randint(-5, 15))
    while detection_circuit_negatif(H, s):
        for u, v in H.items():
            for i in range(len(v)):
                v[i] = (v[i][0], random.randint(-5, 15))

    return graphes, H


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


def ordre_tot(G, s, nbGraphes):
    """
    Retourne un ordre <tot à partir d'un graphe

    Paramètre:
        G : dict(int : list[(int, int)]) - Le graphe
        s: int - la source depuis laquelle exécuter bellman_ford
        nbGraphes : le nombre de graphes aléatoires à générer pour déterminer l'ordre

    Valeur de retour :
        list[int] - L'ordre <tot déterminé
    """
    graphes, H = creation_graphes(G, nbGraphes, s)
    resultats = [Bellman_Ford(graphe, s, G.keys()) for graphe in graphes]

    arb = [t[0] for t in resultats]
    nb_it = [t[1] for t in resultats]

    T = functools.reduce(union, arb)  # Applique la fonction union sur G1 et G2, puis le résultat et G3 ... jusqu'à ce qu'il reste un seul élément
    print(f"{T=}")

    return GloutonFas(T), H

def candidat_source(G, s):
    """
    Vérifie si le sommet s peut être choisi comme source de G

    Paramètres:
        G : dict(int : list[(int, int)]) - Un graphe G
        s: int - le sommet dont on veut vérifier s'il peut être une source
        nbGraphes : le nombre de graphes aléatoires à générer pour déterminer l'ordre

    Valeur de retour :
        bool - True si le sommet peut être une source (a accès à la moitié des sommets) faux sinon
    """
    # On effectue un BFS pour marquer les sommets accessibles depuis le sommet source
    n = len(G)
    visited = [0] * (max(G.keys()) + 1)
    pile = [s]

    while pile:  # On traite des sommets tant que la pile n'est pas vide
        u = pile.pop()

        if not visited[u]:
            visited[u] = 1

            for v in G[u]:  # On ajoute les voisins de u dans la pile
                pile.append(v[0])

    return sum(visited) - 1 >= n - n//2  # Retourne vrai si on a visité plus de la moitié des sommets


def create_graph_random(n, p) :
    G = {i: [] for i in range(1, n+1)}

    for u in range(1, n+1):
        for v in range(1, n+1):
            if u != v and random.random() < p:
                G[u].append((v, random.randint(-5, 15)))

    sources_candidates = [s for s in range(1, n+1) if candidat_source(G, s)]

    return G, random.choice(sources_candidates)


def detection_circuit_negatif(G, s) :
    """
    Detecte grâce a l'algorithme Bellman-Ford si G est un circuit négatif
    """
    n = len(G)
    nb_it = 0
    d = {u: 0 if u == s else sys.maxsize for u in G.keys()}
    # boucle principale
    for i in range(n-1):
        boolean = True
        # u : sommet
        # v : liste des arcs sortant de u avec le poid correspondant
        for u, v in G.items():
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
            nb_it += 1
    
    # verifie si il y a un circuit négatif
    for u, v in G.items() :
        for arcs in v :
            if d[u] + arcs[1] < d[arcs[0]]:
                return True
    
    return False

if __name__ == "__main__":
    # G = exemple2()

    # print(f"{G=}")

    # # Sommet source
    # s = 1

    # # Question 1 / 2
    # arborescence, nb_it = Bellman_Ford(G, s, G.keys())
    # print("arborescence = ", arborescence)
    # print("Nombre iterations = ", nb_it)

    # s = GloutonFas(G)
    # print(f"{s=}")
    
    # # Question 3
    # print(f"\n -- Exemple 3 --\n\n{G=}\n")
    # G1, G2, G3, H = creation_graphes(G)
    
    # # Question 4 / 5
    # tot = ordre_tot(H, 0)
    # print(f"{tot=}\n")
 
    # # Questions 6 / 7
    # s = 4
    # # ordre total
    # arb_tot, nb_it_tot = Bellman_Ford(H, s, tot)
    # # ordre random
    # ordre_rand = list(H.keys())
    # random.shuffle(ordre_rand)
    
    # arb_rand, nb_it_rand = Bellman_Ford(H, s, ordre_rand)
    # print(f"{arb_tot=}\n{nb_it_tot=}\n")
    # print(f"{arb_rand=}\n{nb_it_rand=}\n")
    

    # Question 9
    Nmax = 21
    tab_nb_it_tot = [] # nb itérations pour ordre total
    tab_nb_it_rand=  [] # nb itérations pour ordre random
    tab = [] # sommets

    for i in range(4, Nmax, 2) :
        it_tot = 0
        it_rand = 0
        for j in range(2):
            print(f"{i}/{Nmax}")
            G, s = create_graph_random(i, math.sqrt(i)/i)
            while (detection_circuit_negatif(G, s)) : # si detection de graphe circuit
                G, s = create_graph_random(i, math.sqrt(i)/i)

            # ordre total
            # print("on est la")
            tot, H = ordre_tot(G, s, 4)
            # print("on est la2")
            # print(tot)
            arb_tot, nb_it_tot = Bellman_Ford(H, s, tot)
            # ordre random
            ordre_rand = list(H.keys())
            random.shuffle(ordre_rand)
            arb_rand, nb_it_rand = Bellman_Ford(H, s, ordre_rand)

            it_rand += nb_it_rand
            it_tot += nb_it_tot

        
        # ajout dans les tableaux  
        tab_nb_it_tot.append(it_tot)
        tab_nb_it_rand.append(it_rand)
        tab.append(i)
    
    # comparaison des nombres d'iterations  
    print(tab_nb_it_tot)
    print(tab_nb_it_rand)

    xaxis = list(range(4, Nmax, 2))
    
    # graphique des résultats
    plt.plot(xaxis, tab_nb_it_tot, label="ordre total")
    plt.plot(xaxis, tab_nb_it_rand, label="ordre random")
    # plt.axis([xmin, xmax, ymin, ymax])
    # plt.axis(list(range(4, Nmax, 2)))
    plt.title("Comparaison du nombre d'itérations des deux ordres \n selon la taille du graphe")
    plt.xlabel("Nombre de sommets")
    plt.ylabel("Nombre d'itérations")
    plt.legend()
    plt.show()
        
   
