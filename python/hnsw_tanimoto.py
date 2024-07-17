import numpy as np
import random
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity, ConvertToExplicit
from rdkit.Chem import AllChem
from time import time
from heapq import heappush, heappop, nsmallest
from bisect import insort
from time import time

class HNSW:
    def __init__(self, M, Mmax, efConstruction, mL, fps):
        self.M = M # number of established connections, do i change this parameter (add 1 for each new connection) ?
        self.Mmax = Mmax # maximum number of connections for each element per layer, in this algo Mmax0 = Mmax
        self.efConstruction = efConstruction # size of the dynamic candidate list
        self.mL = mL # normalization factor for level generation
        self.layers = [{} for i in range(100)] # list of layers. Each layer is represented by a dictionary. In each dictionary, the key is a compound and the value is a set of its neighbors at that level
        self.enter_point = None
        self.top_layer = -1
        self.fps = fps
        self.time =[0] * 8

    # insert q
    def insert(self, q):
        temps = time()
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

        ep=[ep]
        for lc in range(min(L, l), -1, -1):
            W = self.search_layer(q, ep, self.efConstruction, lc) # efConstruction nearest elements to q
            neighbors = self.select_neighbors(q, W, self.M, lc) # use efConstruction to select M nearest elements to q 
            self.add_connections(q, neighbors, lc)

            # for each neighbor, remove a connection if there are too many
            for e in neighbors:
                eConn = self.neighborhood(e, lc)
                if len(eConn) > self.Mmax:
                    eNewConn = self.select_neighbors(e, eConn, self.Mmax, lc)
                    self.set_neighborhood(e, eNewConn, lc)
                    for neighbor in eConn:
                        if neighbor not in eNewConn:
                            self.layers[lc][neighbor].remove(e)
            ep = W

        if l > L:
            self.enter_point = q
            self.top_layer = l
        self.time[0] += time()-temps

    # ef closest neighbors to q
    def search_layer(self, q, ep, ef, lc):
        temps = time()
        visited = set()  # set of visited elements for O(1) lookups
        C = []  # min-heap (priority queue) of candidates
        W = []  # max-heap of found nearest neighbors (using negative distances for max-heap behavior)

        # Initial insertion of elements from ep into heaps
        for e in ep:
            distance_e_q = self.distance(e, q)
            heappush(C, (distance_e_q, e))
            heappush(W, (-distance_e_q, e))
            visited.add(e)

        while C:
            c_dist, c_elem = heappop(C)  # Get the element with the minimum distance
            if -W[0][0] < c_dist:
                break  # All elements in W are evaluated

            for e in self.neighborhood(c_elem, lc):
                if e not in visited:
                    distance_e_q = self.distance(e, q)
                    visited.add(e)

                    if len(W) < ef or distance_e_q < -W[0][0]:
                        heappush(C, (distance_e_q, e))
                        heappush(W, (-distance_e_q, e))
                        if len(W) > ef:
                            heappop(W)  # Remove the furthest element

        self.time[1] += time() - temps
        return [e for _, e in nsmallest(ef, W, key=lambda x: -x[0])]

    
    # M nearest elements from C to q
    def select_neighbors(self, q, C, M, lc):
        temps = time()
        result = set(sorted(C, key=lambda e: self.distance(e, q))[:M])
        self.time[2] += time()-temps
        return result

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
        temps = time()
        W = []
        ep = self.enter_point
        L = self.top_layer
        for lc in range(L, 0, -1):
            W = self.search_layer(q, [ep], 1, lc)
            ep = W[0]
        W = self.search_layer(q, [ep], ef, 0)
        result = set(sorted(W, key=lambda e: self.distance(e, q))[1:K+1])
        self.time[3] += time()-temps
        return result

    def distance(self, mol1, mol2):
        temps = time()
        res = 1 - TanimotoSimilarity(self.fps[mol1], self.fps[mol2])
        self.time[4] += time()-temps
        return res

    # retrieve the neighborhood of an element at a given layer
    def neighborhood(self, element, lc):
        temps = time()
        if lc < self.top_layer:
            result = self.layers[lc].get(element, set())
            self.time[5] += time()-temps
            return result
        return set()

    # add connections between an element and its neighbors in a specified layer
    def add_connections(self, q, neighbors, lc):
        temps = time()
        for neighbor in neighbors:
            self.layers[lc][q].add(neighbor)
            self.layers[lc][neighbor].add(q)
        self.time[6] += time()-temps

    # set the neighborhood of an element in a specified layer
    def set_neighborhood(self, element, neighborhood, lc):
        temps = time()
        self.layers[lc][element] = neighborhood
        self.time[7] += time()-temps




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