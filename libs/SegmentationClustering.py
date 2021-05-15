import numpy as np
import matplotlib.pyplot as plt
import cv2
np.random.seed(42)

# KMeans Algorithm

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans():

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            print(f"len cluster: {cluster_idx} = {len(cluster)}")
            cluster_mean = np.mean(self.X[cluster], axis=0)
            print(f"cluster_mean {cluster_idx}: {cluster_mean}")
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()
    def cent(self):
        return self.centroids

def apply_k_means(source, k=5, max_iter=100):
    """Segment image using K-means

    Args:
        source (nd.array): BGR image to be segmented
        k (int, optional): Number of clusters. Defaults to 5.
        max_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        segmented_image (nd.array): image segmented
        labels (nd.array): labels of every point in image
    """
    # convert to RGB
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # reshape image to points
    pixel_values = source.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # run k-means algorithm
    model = KMeans(K=k, max_iters=max_iter)
    y_pred = model.predict(pixel_values)

    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)

    # flatten labels and get segmented image
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(source.shape)

    return segmented_image, labels


def apply_region_growing(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)

    return src


def apply_agglomerative(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)

    return src


def apply_mean_shift(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)

    return src