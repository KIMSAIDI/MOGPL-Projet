# -*- coding: utf-8 -*-
import math
import sys

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
    # nb_it : nombre d'itération nécessaire avant la convergence de l'algorithme
    nb_it = 0 
    # d : liste representant les longueurs des chemins entre s et les autres sommets
    d = dict()
    # on initialise pour tout les sommets dans G, la longueur = infini et pour s = 0
    for u in G.keys() :
        if u == s :
            d[u] = 0
        else :
            d[u] = sys.maxsize
    
    # boucle principal
    for i in range(n-1) :
        # boolean qui va nous permettre de savoir si l'algorithme à convergé
        boolean = True
        # u : sommet
        # v : liste des arcs sortant de u avec le poid correspondant
        for u, v in G.items() :
            # arcs : tuple(int, int) ; arcs[0] : sommet, arcs[1] : poids
            for arcs in v :
                if d[u] + arcs[1] < d[arcs[0]] :
                    # si on rentre dans cette condition : l'algorithme n'a toujours pas convergé
                    boolean = False
                    # on update la valeur de la longueur
                    d[arcs[0]] = d[u] + arcs[1]
        
        if boolean : # l'algorithme a convergé 
            break
        else :
            nb_it+=1          
        
    
    return d, nb_it


# Exemple de graphe
G = {
    1: [(2, 3), (3, 5)],
    2: [(3, -2)],
    3: [(4, 2)],
    4: []
}

# Sommet source
s = 1

d, nb_it = Bellman_Ford(G, s)
print("d = ", d)
print("Nombre iterations = ", nb_it)



        
