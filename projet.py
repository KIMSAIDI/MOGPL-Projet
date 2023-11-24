# -*- coding: utf-8 -*-
import sys

from exemples import *


# Question 1
def Bellman_Ford(G, s):
    """
    Renvoie l'algorithme de Bellman_Ford.
    Le graphe ne contient aucun circuit négatif (d'après énoncé).

    Paramètres:
        G : dict(int : list[(int, int)]) -
            Graphe, représenté sous la forme d'un dictionnaire. 
            Les keys (int) sont tous les sommets du graphes. 
            Les values (list[(int, int)]) sont tous les sommets sortant du sommet et sont representés sous la formes d'un tuples avec comme deuxième element le poid associé
        
        s : int - sommet
    """
    # n : nombre de sommet dans G
    n = len(G)
    # nb_it : nombre d'itérations nécessaire avant la convergence de l'algorithme
    nb_it = 0
    # d : liste représentant les longueurs des chemins entre s et les autres sommets
    d = dict()
    # on initialise pour tous les sommets dans G, la longueur = infini et pour s = 0
    for u in G.keys():
        if u == s:
            d[u] = 0
        else:
            d[u] = sys.maxsize

    # Luc : On peut aussi faire des compréhensions de dictionnaires
    # d = {u: 0 if u == s else sys.maxsize for u in G.keys()}

    # boucle principale
    for i in range(n - 1):
        # boolean qui va nous permettre de savoir si l'algorithme a convergé
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

    return d, nb_it


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


if __name__ == "__main__":
    G = exemple1()

    # Sommet source
    s = 1

    d, nb_it = Bellman_Ford(G, s)
    print("d = ", d)
    print("Nombre iterations = ", nb_it)

    s = GloutonFas(G)
    print(f"{s=}")
