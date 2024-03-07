from scipy.cluster.hierarchy import linkage 
import numpy as np


RANDOM_STATE=42
np.random.seed(RANDOM_STATE)

class HierarchicalLsh:
    
    def __init__(self, num_levels, maximum_sample_size=1_000):
        self.num_levels = num_levels
        self._maximum_sample_size = maximum_sample_size

    def fit(self, data):
        leaves = self._select_leaf_data(data, self._maximum_sample_size)
        self.clustering = HierarchicalClustering(leaves, method='ward')  # TODO: try other methods of linking

    def hash_vector(self, vector):
        return self.clustering.calculate_traversal_hash(vector, self.num_levels)

    def to_string(self):
        return f"HierarchicalLsh (num_levels={self.num_levels})"
    
    def _select_leaf_data(self, data, maximum_sample_size):
        if maximum_sample_size < data.shape[0]:
            print(f"Using a random subset of {maximum_sample_size} samples for hierarchical clustering")
            np.random.seed(RANDOM_STATE)
            shuffled_data = np.random.permutation(data)
            return shuffled_data[:maximum_sample_size]
        print(f"Using full {data.shape[0]} data items as leaves in hierarchical clustering")
        return data
    
class HierarchicalClustering:

    BIT_AS_0 = '0'
    BIT_AS_1 = '1'

    def __init__(self, data, method='ward'):
        self.leaves = data
        # clusters are in z format. Each cluster has 4 values: 2 children indexes, children distance, total leaves
        # the children can be leaves (i.e. data_subset index), or other clusters (i.e. index of cluster + len(data_subset), for uniqueness)
        # clusters' ordering is based on iterations of cluster merging, starting from leave pairs, and ending with the root cluster 
        # therefore the length of z is always n-1, where n is the number of leaves
        self.clusters = linkage(data, method=method)

    def find_nearest_leaf(self, vector):
        root_cluster_id = self._get_root_cluster_id()
        leaf, _, _ = self._traverse_to_nearest_leaf(root_cluster_id, vector)
        return leaf

    def calculate_traversal_hash(self, vector, num_levels):
        root_cluster_id = self._get_root_cluster_id()
        _, _, hash = self._traverse_to_nearest_leaf(root_cluster_id, vector)
        return hash[:num_levels]

    def _traverse_to_nearest_leaf(self, node_id, vector):
        if self._is_leaf(node_id): 
            leaf = self.leaves[node_id]
            return (leaf, np.linalg.norm(vector - leaf), '')

        cluster = self._get_cluster(node_id)
        left_leaf, left_distance, left_hash = self._traverse_to_nearest_leaf(int(cluster[0]), vector)
        right_leaf, right_distance, right_hash = self._traverse_to_nearest_leaf(int(cluster[1]), vector)

        if left_distance < right_distance:
            return (left_leaf, left_distance, self.BIT_AS_0 + left_hash)
        return (right_leaf, right_distance, self.BIT_AS_1 + right_hash)

    def _get_root_cluster_id(self):
        return len(self.leaves) + len(self.clusters) - 1

    def _get_cluster(self, node_id):
        cluster_id = node_id - len(self.leaves)
        if cluster_id < 0:
            raise ValueError(f"Node {node_id} is not a cluster")
        return self.clusters[cluster_id]

    def _is_leaf(self, node_id):
        return node_id < len(self.leaves)