from mnist import MNIST
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


# Wczytaj dane MNIST
mndata = MNIST() 
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Przekształcamy dane na macierze NumPy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Implementacja KNN
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

knn = KNN(k=3)

# trening modelu na danych treningowych
knn.fit(X_train, y_train)

# Przewidujemy klasy/etykiety dla 100 pierwszych
predictions = knn.predict(X_test[:100])

#Wynik
print(predictions)
accuracy = accuracy_score(y_test[:100], predictions)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")



