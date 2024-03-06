import numpy as np


class SuperBitLsh:
    
    BIT_AS_0 = 0
    BIT_AS_1 = 1
    BASE_TWO = 2
    
    def __init__(self, hash_length, num_bits_per_batch):
        self.hash_length = hash_length
        self.num_bits_per_batch = num_bits_per_batch

    def _generate_random_vectors(self, dimension_count):
        return np.random.randn(self.hash_length, dimension_count)

    def _gram_schmidt(self, vectors):
        for i in range(1, len(vectors)):
            for j in range(i):
                proj = np.dot(vectors[j], vectors[i]) * vectors[j]
                vectors[i] -= proj
        # Normalize the orthogonalized vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors

    def to_string(self):
        return f"SuperBitLsh(hash_length={self.hash_length}, num_bits_per_batch={self.num_bits_per_batch})"

    def fit(self, X):
        dimension_count = X.shape[1]
        self.random_vectors = self._generate_random_vectors(dimension_count)
        pass

    def hash_vector(self, vector):
        hash_values = []
        for i in range(0, self.hash_length, self.num_bits_per_batch):
            batch = self.random_vectors[i:i + self.num_bits_per_batch]
            batch = self._gram_schmidt(batch) 
            projections = np.dot(batch, vector)
            hashes_as_bits = np.where(projections > 0, self.BIT_AS_1, self.BIT_AS_0)
            hash_values.extend(hashes_as_bits)
        return self._hash_bits_to_integer(hash_values)
    
    def _hash_bits_to_integer(self, hash_bits):
        hash_bits_string = ''.join(str(bit) for bit in hash_bits)
        return int(hash_bits_string, self.BASE_TWO)