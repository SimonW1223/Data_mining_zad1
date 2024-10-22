import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Liczymy odległość euklidesową
            differences = self.X_train - x
            distances = np.sqrt(np.sum(differences**2, axis=1))
            # Znajdujemy k najbliższych sąsiadów
            sorted_indices = np.argsort(distances)
            k_indices = sorted_indices[:self.k] 
            k_nearest_labels = []
            for i in k_indices:
                k_nearest_labels.append(self.y_train[i])
             # Najczęściej występująca etykieta
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions