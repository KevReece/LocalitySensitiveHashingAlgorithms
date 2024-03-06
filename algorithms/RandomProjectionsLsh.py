import numpy as np


RANDOM_STATE=42

class RandomProjectionsLsh:
    '''This LSH algorithm uses randomly generates some vectors. 
    Each of these vectors to determine a bit in the hash. 
    The input vector is projected into each vector as one dimension, and the sign becomes the bit.'''
    
    BIT_AS_0 = 0
    BIT_AS_1 = 1
    BASE_TWO = 2

    def __init__(self, hash_length):
        self.hash_length = hash_length

    def fit(self, X):
        dimensions_count = X.shape[1]
        self.random_vectors = self._generate_random_vectors(self.hash_length, dimensions_count)
        pass

    def hash_vector(self, vector):
        projections = np.dot(self.random_vectors, vector)
        hashes_as_bits = np.where(projections > 0, self.BIT_AS_1, self.BIT_AS_0)
        return self._hash_bits_to_integer(hashes_as_bits)
    
    def to_string(self):
        return f'RandomProjectionsLsh (hash_length={self.hash_length})'
    
    def _hash_bits_to_integer(self, hash_bits):
        hash_bits_string = ''.join(str(bit) for bit in hash_bits)
        return int(hash_bits_string, self.BASE_TWO)
    
    def _generate_random_vectors(self, hash_length, dimensions_count):
        return np.random.randn(hash_length, dimensions_count)