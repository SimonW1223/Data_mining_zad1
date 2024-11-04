import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from algorythm import KNN

# Wczytaj zbiór danych Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Przetwarzanie danych
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

# Kodowanie cechy 'Sex'
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Podział na cechy i zmienną docelową
X = data.drop('Survived', axis=1)
y = data['Survived']

# Standaryzacja cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział danych na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testowanie różnych wartości k
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

# Inicjalizacja modelu KNN z najlepszym k (np. k=best_k)
knn = KNN(k=best_k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Wyniki
results = pd.DataFrame({
    'Predicted': predictions,
})
print(results.head(20))

# Obliczenie końcowej dokładności
accuracy = accuracy_score(y_test, predictions)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")