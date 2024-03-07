from scipy.cluster.hierarchy import linkage 
import numpy as np


RANDOM_STATE=42

class HierarchicalHyperplaneLsh:
    '''This LSH algorithm uses a hierarchical clustering to create a tree of hyperplanes.'''
    
    def __init__(self, num_levels, maximum_sample_size=10_000):
        self.num_levels = num_levels
        self._maximum_sample_size = maximum_sample_size

    def fit(self, data):
        leaves = self._select_leaf_data(data, self._maximum_sample_size)
        self.clustering = HierarchicalHyperplaneClustering(leaves, self.num_levels)

    def hash_vector(self, vector):
        return self.clustering.calculate_traversal_hash(vector)

    def to_string(self):
        return f"HierarchicalHyperplaneLsh(num_levels={self.num_levels})"

    def _select_leaf_data(self, data, maximum_sample_size):
        if maximum_sample_size < data.shape[0]:
            print(f"Using a random subset of {maximum_sample_size} samples for hierarchical clustering")
            np.random.seed(RANDOM_STATE)
            shuffled_data = np.random.permutation(data)
            return shuffled_data[:maximum_sample_size]
        print(f"Using full {data.shape[0]} data items as leaves in hierarchical clustering")
        return data

class ClusterValue:
    LEFT = 0
    RIGHT = 1
    DISTANCE = 2
    TOTAL_LEAVES = 3
    PARENT = 4
    LEVEL = 5

class HierarchicalHyperplaneClustering:

    BIT_AS_0 = '0'
    BIT_AS_1 = '1'

    def __init__(self, leaves, num_levels):
        self.leaves = leaves
        self._count_leaves = len(leaves)
        # clusters are in z format. Each cluster has 4 values: 2 children indexes, children distance, total leaves
        # the children can be leaves (i.e. data_subset index), or other clusters (i.e. index of cluster + len(data_subset), for uniqueness)
        # clusters' ordering is based on iterations of cluster merging, starting from leave pairs, and ending with the root cluster 
        # therefore the length of z is always n-1, where n is the number of leaves
        print(f"Building hierarchy over {len(leaves)} leaves")
        self.clusters = linkage(leaves, method='ward')
        self.num_levels = num_levels
        self._enrich_clusters()
        print(f'{num_levels} hierarchical hyperplanes calculated')

    def calculate_traversal_hash(self, vector):
        root_node_id = self._get_root_node_id()
        hash = self._traverse_down_nodes_for_hash(root_node_id, vector)
        return hash

    def _enrich_clusters(self):
        self.clusters = np.pad(self.clusters,((0,0),(0,2)))
        self.cluster_hyperplanes = {}
        print(f"Enriching clusters with parents and levels and hyperplanes")
        self._traverse_enriching_clusters_with_parents_and_levels_and_hyperplanes(self._get_root_node_id())

    def _traverse_enriching_clusters_with_parents_and_levels_and_hyperplanes(self, node_id, parent_id=None, level=0):
        if self._is_leaf(node_id):
            return self._get_leaf(node_id), 0
        cluster = self._get_cluster(node_id)
        cluster[ClusterValue.PARENT] = parent_id
        cluster[ClusterValue.LEVEL] = level
        left_mean, left_standard_deviation = self._traverse_enriching_clusters_with_parents_and_levels_and_hyperplanes(cluster[ClusterValue.LEFT], node_id, level+1)
        right_mean, right_standard_deviation = self._traverse_enriching_clusters_with_parents_and_levels_and_hyperplanes(cluster[ClusterValue.RIGHT], node_id, level+1)
        cluster_mean, cluster_standard_deviation = self._generalized_mean_std(left_mean, right_mean, left_standard_deviation, right_standard_deviation)
        right_direction = right_mean - left_mean
        unit_right_direction = right_direction / np.linalg.norm(right_direction)
        self.cluster_hyperplanes[int(node_id)] = (cluster_mean, unit_right_direction)
        return cluster_mean, cluster_standard_deviation

    def _traverse_down_nodes_for_hash(self, node_id, vector):
        if self._is_leaf(node_id): 
            raise ValueError(f"Node {node_id} is a leaf")

        cluster = self._get_cluster(node_id)
        cluster_hyperplane_centre, cluster_hyperplane_direction = self.cluster_hyperplanes[int(node_id)]

        is_left = np.dot(vector - cluster_hyperplane_centre, cluster_hyperplane_direction) < 0
        is_penultimate_level = int(cluster[ClusterValue.LEVEL]) == self.num_levels-1

        if is_penultimate_level:
            return self.BIT_AS_0 if is_left else self.BIT_AS_1
        
        if is_left:
            if self._is_leaf(cluster[ClusterValue.LEFT]): 
                return self.BIT_AS_0
            return self.BIT_AS_0 + self._traverse_down_nodes_for_hash(cluster[ClusterValue.LEFT], vector)
        
        if self._is_leaf(cluster[ClusterValue.RIGHT]):
            return self.BIT_AS_1
        return self.BIT_AS_1 + self._traverse_down_nodes_for_hash(cluster[ClusterValue.RIGHT], vector)
    
    def _generalized_mean_std(self, mean1, mean2, standard_deviation1, standard_deviation2):
        MIN_STANDARD_DEVIATION = 1e-6 # to avoid division by zero in the case of leaves
        inverse_squared_standard_deviation1 = 1 / max(standard_deviation1, MIN_STANDARD_DEVIATION)**2
        inverse_squared_standard_deviation2 = 1 / max(standard_deviation2, MIN_STANDARD_DEVIATION)**2
        weights = np.array([inverse_squared_standard_deviation1, inverse_squared_standard_deviation2])
        weights /= np.sum(weights)

        mean = np.average(np.vstack([mean1, mean2]), axis=0, weights=weights)

        deviation1 = np.linalg.norm(mean1 - mean)
        deviation2 = np.linalg.norm(mean2 - mean)
        standard_deviation = (deviation1 + deviation2)/2

        return mean, standard_deviation

    def _get_root_node_id(self):
        return self._count_leaves + len(self.clusters) - 1

    def _get_cluster(self, node_id):
        cluster_id = node_id - self._count_leaves
        if cluster_id < 0:
            raise ValueError(f"Node {node_id} is not a cluster")
        return self.clusters[int(cluster_id)]

    def _is_leaf(self, node_id):
        return node_id < self._count_leaves
    
    def _get_leaf(self, node_id):
        return self.leaves[int(node_id)]