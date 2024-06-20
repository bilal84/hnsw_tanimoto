from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity

class NOHNSW:
    def __init__(self, fps):
        self.fps = fps
    
    def distance(self, mol1, mol2):
        res = 1 - TanimotoSimilarity(self.fps[mol1], self.fps[mol2])
        return res
    
    def k_nearest_neighbors(self, target_index, k=100):
        distances = []
        for i in range(len(self.fps)):
            if i != target_index:
                dist = self.distance(target_index, i)
                distances.append((i, dist))
        
        # sort distances by ascending order
        distances.sort(key=lambda x: x[1])
        
        # take the k nearest neighbors
        nearest_neighbors = distances[:k]
        return [nearest_neighbors[i][0] for i in range(k)]