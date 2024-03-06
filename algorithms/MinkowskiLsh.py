import numpy as np


RANDOM_STATE=42
np.random.seed(RANDOM_STATE)

class MinkowskiLsh:
    '''This LSH algorithm has a number of random vectors, then for an input vector it:
    1. choses the closest random vector
    2. projects the input vector onto it as one dimension
    3. this float value is the hash, but is scaled, as desired for bucketing, then floored to the int hash.'''

    def __init__(self, dimensions_count, bucket_width, count_hash_tables):
        self.dimensions_count = dimensions_count
        self.bucket_width = bucket_width
        self.random_vectors = self._generate_random_vectors(count_hash_tables)

    def _generate_random_vectors(self, count_hash_tables):
        random_vectors = []
        for _ in range(count_hash_tables):
            random_vectors.append(self._generate_random_vector())
        return random_vectors
    
    def _generate_random_vector(self):
        raise NotImplementedError("abstract method: _generate_random_vector")

    def _get_vector_distance(self, vector, other_vector):
        raise NotImplementedError("abstract method: _get_vector_distance")

    def _hash_single_vector(self, vector, random_vector):
        projection = np.dot(vector, random_vector)
        hash_value = int(projection // self.bucket_width)
        return hash_value
    
    def fit(self, _):
        # MinkowskiLsh is data independent, so we don't need to fit it
        pass

    def hash_vector(self, vector):
        hash = 0
        closest_random_vector_distance = float('inf')
        for random_vector in self.random_vectors:
            distance = self._get_vector_distance(vector, random_vector) 
            if distance < closest_random_vector_distance:
                closest_random_vector_distance = distance
                hash = self._hash_single_vector(vector, random_vector)
        return hash

    def to_string(self):
        raise NotImplementedError("abstract method: to_string")
    
class L2Lsh(MinkowskiLsh):
    def _generate_random_vector(self):
        return np.random.randn(self.dimensions_count) 
    
    def _get_vector_distance(self, vector, other_vector):
        return np.linalg.norm(other_vector - vector) 
    
    def to_string(self):
        return f'L2Lsh (dimensions_count={self.dimensions_count}, bucket_width={self.bucket_width}, count_hash_tables={len(self.random_vectors)}'

class L1Lsh(MinkowskiLsh):
    def _generate_random_vector(self):
        return np.random.choice([-1, 1], size=self.dimensions_count)
    
    def _get_vector_distance(self, vector, other_vector):
        return np.sum(np.abs(other_vector - vector))
    
    def to_string(self):
        return f'L1Lsh (dimensions_count={self.dimensions_count}, bucket_width={self.bucket_width}, count_hash_tables={len(self.random_vectors)}'