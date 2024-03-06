import numpy as np
from sklearn.decomposition import PCA


RANDOM_STATE=42
np.random.seed(RANDOM_STATE)

class PcaMinkowskiLsh:
    '''Mashup of PcaBasedLsh and MinkowskiLsh.
    Gives hash based on distance to nearest PCA component'''

    def __init__(self, bucket_width, count_hash_tables):
        self.bucket_width = bucket_width
        self.count_hash_tables = count_hash_tables

    def _get_vector_distance(self, vector, other_vector):
        raise NotImplementedError("abstract method: _get_vector_distance")

    def _hash_single_vector(self, vector, pca_component):
        projection = np.dot(vector, pca_component)
        hash_value = int(projection // self.bucket_width)
        return hash_value
    
    def fit(self, X):
        count_dimensions = X.shape[1]
        if self.count_hash_tables > count_dimensions:
            print(f"count_hash_tables ({self.count_hash_tables}) should be less than or equal to count_dimensions ({count_dimensions}). Capping.")
        pca_components = min(self.count_hash_tables, count_dimensions)
        self.pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
        self.pca.fit(X)

    def hash_vector(self, vector):
        hash = 0
        closest_random_vector_distance = float('inf')
        for pcs_component in self.pca.components_:
            distance = self._get_vector_distance(vector, pcs_component) 
            if distance < closest_random_vector_distance:
                closest_random_vector_distance = distance
                hash = self._hash_single_vector(vector, pcs_component)
        return hash
    
    def to_string(self):
        raise NotImplementedError("abstract method: to_string")

class PcaL2Lsh(PcaMinkowskiLsh):
    def _get_vector_distance(self, vector, other_vector):
        return np.linalg.norm(other_vector - vector) 
    
    def to_string(self):
        return f'PcaL2Lsh (bucket_width={self.bucket_width}, count_hash_tables={self.count_hash_tables})'

class PcaL1Lsh(PcaMinkowskiLsh):
    def _get_vector_distance(self, vector, other_vector):
        return np.sum(np.abs(other_vector - vector))
    
    def to_string(self):
        return f'PcaL1Lsh (bucket_width={self.bucket_width}, count_hash_tables={self.count_hash_tables})'