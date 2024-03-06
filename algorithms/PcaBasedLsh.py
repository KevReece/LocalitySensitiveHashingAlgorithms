from sklearn.decomposition import PCA
import numpy as np


RANDOM_STATE=42

class PcaBasedLsh:
    '''This LSH algorithm uses PCA to determine some vectors of importance for the hash function. 
    Each of these vectors to determine a bit in the hash. 
    The input vector is projected into each vector as one dimension, and the sign becomes the bit.'''
    
    BIT_AS_0 = 0
    BIT_AS_1 = 1
    BASE_TWO = 2

    def __init__(self, hash_length):
        self.pca = PCA(n_components=hash_length, random_state=RANDOM_STATE)

    def fit(self, X):
        self.pca.fit(X)

    def hash_vector(self, vector):
        # Each component is a set of dimensions which define a hyperplane
        projections = np.dot(self.pca.components_, vector)
        hashes_as_bits = np.where(projections > 0, self.BIT_AS_1, self.BIT_AS_0)
        return self._hash_bits_to_integer(hashes_as_bits)
    
    def to_string(self):
        return f'PcaBasedLsh (hash_length={self.pca.n_components})'
    
    def _hash_bits_to_integer(self, hash_bits):
        hash_bits_string = ''.join(str(bit) for bit in hash_bits)
        return int(hash_bits_string, self.BASE_TWO)