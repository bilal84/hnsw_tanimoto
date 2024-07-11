from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.DataStructs import ExplicitBitVect

class NOHNSW:
    def __init__(self, fps):
        self.fps = fps

    def distance(self, mol1, mol2):
        # Convert lists to ExplicitBitVect
        bv1 = ExplicitBitVect(len(self.fps[mol1]))
        bv2 = ExplicitBitVect(len(self.fps[mol2]))
        
        for i, bit in enumerate(self.fps[mol1]):
            if bit:
                bv1.SetBit(i)
        
        for i, bit in enumerate(self.fps[mol2]):
            if bit:
                bv2.SetBit(i)
        
        res = 1 - TanimotoSimilarity(bv1, bv2)
        return res

    def k_nearest_neighbors(self, target_index, k):
        distances = []
        for i in range(len(self.fps)):
            if i != target_index:
                dist = self.distance(target_index, i)
                distances.append((dist, i))
        distances.sort()
        return [index for dist, index in distances[:k]]
