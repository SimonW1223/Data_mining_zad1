from mnist import MNIST
import numpy as np
from sklearn.metrics import accuracy_score
from algorythm import KNN
import matplotlib.pyplot as plt


# Wczytaj dane MNIST
mndata = MNIST() 
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Przekształcamy dane na macierze NumPy
X_train = np.array(X_train)[:1000]
y_train = np.array(y_train)[:1000]
X_test = np.array(X_test)[:200]
y_test = np.array(y_test)[:200]


k_values = range(1, 10)
k_scores = []

for k in k_values:
    knn = KNN(k=k)  # Inicjalizacja KNN z bieżącym k
    knn.fit(X_train, y_train)  # Trenuj model na danych treningowych
    predictions = knn.predict(X_test)  # Przewiduj dla zbioru testowego
    accuracy = accuracy_score(y_test, predictions)  # Oblicz dokładność
    k_scores.append(accuracy)

# Znalezienie najlepszego k
best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)

print(f"The best value of k is {best_k} with an accuracy of {best_score:.4f}")

# Rysowanie wykresu dokładności dla różnych wartości k
plt.plot(k_values, k_scores, marker='o')
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy on Test Set')
plt.title('Finding the Best K for KNN on Titanic Dataset')
plt.grid()
plt.show()

knn = KNN(k=best_k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test[:1000])

#Wyniki
print(predictions)
accuracy = accuracy_score(y_test[:1000], predictions)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")