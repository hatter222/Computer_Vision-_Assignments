import cv2
import numpy as np
import pickle
import os
from decimal import  Decimal,ROUND_UP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from face_detector import FaceDetector



# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = face - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.reshape(face, (1, 3, 224, 224))
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=8, max_distance=0.2, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))


        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    def update(self, face, label):
        # extracts its embedding, and stores it as a training sample in the gallery
        temp = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [temp], axis=0)
        self.labels.append(label)
        return None

    # ToDo
    def predict(self, face):
        # identity is determined by taking the majority of identities of the k nearest neighbors
        model = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        embedding = self.facenet.predict(face).reshape(1, -1)
        #print(embedding.shape)
        labels = np.array(self.labels)
        model.fit(self.embeddings, labels)
        predictions = model.predict(embedding)
        dist, idx = model.kneighbors(embedding, 11)
        # print(dist)
        prob = model.predict_proba(embedding)[0]
        #print(prob)
        classes = model.classes_
        n_class = len(classes)
        #print(n_class)
        class_dist = np.ones(n_class) * 100000
        #print(class_dist.shape)

        for i in idx[0]:
            label = labels[i]
            idx_cl = np.where(classes == label)[0][0]
            if class_dist[idx_cl] > dist[0][idx_cl]:
                class_dist[idx_cl] = dist[0][idx_cl]
        # to find the matching
        idx_cl = np.where(classes == predictions)[0][0]
        # openset protocol
        if prob[idx_cl] < self.min_prob and class_dist[idx_cl] > self.max_distance:
            predictions = "Unknown"
        return predictions, prob[idx_cl], class_dist[idx_cl]


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=5, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        #without identity
        temp = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings,[temp], axis=0)
        return None

    # ToDo
    def fit(self):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.embeddings)
        # calculate clusters and assign each face to the closest
        # recalculate cluster centers
        self.cluster_center = kmeans.cluster_centers_
        #print(self.cluster_center)
        self.cluster_membership = kmeans.labels_
        return None

    # ToDo
    def predict(self, face):

        embedding = self.facenet.predict(face)
        dist = list()
        for i in range(0, self.num_clusters):
            temp = np.linalg.norm(self.cluster_center[i]-embedding)      # bestMatch = min(euclideanDistance)
            temp = Decimal(Decimal(temp).quantize(Decimal('0.1'),ROUND_UP)) # for significant precision
            dist.append(float(temp)) # from decimal to float
        idx = dist.index(min(dist))
        # return bestMatch and distribution
        return idx, dist









