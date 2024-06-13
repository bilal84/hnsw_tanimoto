import numpy as np
import random
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity, ConvertToExplicit
from rdkit.Chem import AllChem

class HNSW:
    def __init__(self, M, Mmax, efConstruction, mL, molecules, fps):
        self.M = M # number of established connections, do i change this parameter (add 1 for each new connection) ?
        self.Mmax = Mmax # maximum number of connections for each element per layer, in this algo Mmax0 = Mmax
        self.efConstruction = efConstruction # size of the dynamic candidate list
        self.mL = mL # normalization factor for level generation
        self.layers = [{} for i in range(1000)] # list of layers. Each layer is represented by a dictionary. In each dictionary, the key is a compound and the value is a set of its neighbors at that level
        self.enter_point = None
        self.top_layer = -1
        self.molecules = molecules
        self.fps = fps

    # insert q
    def insert(self, q):
        W = []
        ep = self.enter_point
        L = self.top_layer
        l = int(-np.log(random.uniform(0, 1)) * self.mL)
        #print(l)

        # add q to all layers up to top_layer
        for lc in range(l + 1):
            self.layers[lc][q] = set()

        # case of the first inserted element
        if ep is None:
            self.enter_point = q
            self.top_layer = l
            return

        # at each level, we look for the nearest element to q
        for lc in range(L, l, -1):
            W = self.search_layer(q, [ep], 1, lc)
            ep = W[0]

        # 
        ep=[ep]
        for lc in range(min(L, l), -1, -1):
            W = self.search_layer(q, ep, self.efConstruction, lc) # efConstruction nearest elements to q
            neighbors = self.select_neighbors_heuristic(q, W, self.M, lc) # use efConstruction to select M nearest elements to q 
            self.add_connections(q, neighbors, lc)

            # for each neighbor, remove a connection if there are too many
            for e in neighbors:
                eConn = self.neighborhood(e, lc)
                if len(eConn) > self.Mmax:
                    eNewConn = self.select_neighbors_heuristic(e, eConn, self.Mmax, lc)
                    self.set_neighborhood(e, eNewConn, lc)
                    for neighbor in eConn:
                        if neighbor not in eNewConn:
                            self.layers[lc][neighbor].remove(e)
            ep = W

        if l > L:
            self.enter_point = q
            self.top_layer = l


    # ef closest neighbors to q
    def search_layer(self, q, ep, ef, lc):
        v = set(ep) # set of visited elements
        C = set(ep) # set of candidates
        W = [ep[i] for i in range(len(ep))] # dynamic list of found nearest neighbors

        # decide if we have to search for the max/min
        search_c = True
        search_f = True

        while len(C) > 0:
            if search_c:
                c = min(C, key=lambda e: self.distance(e, q))
            C.remove(c) # extract nearest element from C to q
            search_c = True

            if search_f:
                f = max(W, key=lambda e: self.distance(e, q)) # get furthest element from W to q
                search_f = False

            if self.distance(c, q) > self.distance(f, q):
                break # all elements in W are evaluated

            distance_f_q = self.distance(f, q)
            for e in self.neighborhood(c, lc): # update C and W
                if e not in v:
                    v.add(e)

                    if search_f:
                        f = max(W, key=lambda e: self.distance(e, q))

                    distance_e_q = self.distance(e, q)
                    if distance_e_q < distance_f_q or len(W) < ef:
                        C.add(e)
                        if distance_e_q < self.distance(c, q):
                            c = e
                            search_c = False # c is still the nearest element from C to q, and it is not the element which was removed, so we don't have to search for the min again

                        W.append(e)
                        if self.distance(e,q) >= distance_f_q:
                            f = e # f is still the furthest element from W to q

                        if len(W) > ef:
                            W.remove(f)
                            search_f = True # we removed f, so we have to search the max. If we dont removed it, it is still the max, so we don't have to search for the max again

        return W
    
    # M nearest elements from C to q
    def select_neighbors(self, q, C, M, lc):
        return set(sorted(C, key=lambda e: self.distance(e, q))[:M])

    # M nearest elements from C to q : other version
    def select_neighbors_heuristic(self, q, C, M, lc, extendCandidates=False, keepPrunedConnections=True):
        R = set()
        W = C.copy() # working queue for the candidates

        if extendCandidates: # extend candidates by their neighbors
            for e in C:
                for eadj in self.neighborhood(e, lc):
                    if eadj not in W:
                        W.add(eadj)

        Wd = set()
        while len(W) > 0 and len(R) < M:
            e = min(W, key=lambda e: self.distance(e, q))
            W.remove(e)
            if all(self.distance(e, q) <= self.distance(r, q) for r in R):
                R.add(e)
            else:
                Wd.add(e)

        if keepPrunedConnections:
            while len(Wd) > 0 and len(R) < M:
                e = min(Wd, key=lambda e: self.distance(e, q))
                Wd.remove(e)
                R.add(e)

        return R


    # K nearest elements to q
    def k_nn_search(self, q, K, ef):
        W = []
        ep = self.enter_point
        L = self.top_layer
        for lc in range(L, 0, -1):
            W = self.search_layer(q, [ep], 1, lc)
            ep = W[0]
        W = self.search_layer(q, [ep], ef, 0)

        return set(sorted(W, key=lambda e: self.distance(e, q))[:K])

    def distance(self, mol1, mol2):
        res = 1 - TanimotoSimilarity(self.fps[mol1], self.fps[mol2])
        return res

    # retrieve the neighborhood of an element at a given layer
    def neighborhood(self, element, lc):
        if lc < self.top_layer:
            return self.layers[lc].get(element, set())
        return set()

    # add connections between an element and its neighbors in a specified layer
    def add_connections(self, q, neighbors, lc):
        for neighbor in neighbors:
            self.layers[lc][q].add(neighbor)
            self.layers[lc][neighbor].add(q)

    # set the neighborhood of an element in a specified layer
    def set_neighborhood(self, element, neighborhood, lc):
        self.layers[lc][element] = neighborhood




# basic version of algorithm 2

# def search_layer(self, q, ep, ef, lc):
#         v = [ep[i] for i in range(len(ep))]
#         C = [ep[i] for i in range(len(ep))]
#         W = [ep[i] for i in range(len(ep))]

#         #print(ep)

#         while len(C) > 0:
#             c = min(C, key=lambda e: self.distance(e, q))
#             C.remove(c)
#             f = max(W, key=lambda e: self.distance(e, q))

#             if self.distance(c, q) > self.distance(f, q):
#                 break

#             for e in self.neighborhood(c, lc):
#                 if e not in v:
#                     v.append(e)
#                     f = max(W, key=lambda e: self.distance(e, q))
#                     if self.distance(e, q) < self.distance(f, q) or len(W) < ef:
#                         C.append(e)
#                         W.append(e)
#                         if len(W) > ef:
#                             W.remove(f)

#         return W