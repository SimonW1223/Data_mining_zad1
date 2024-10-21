from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

# Inicjalizacja MNIST
mndata = MNIST() 

# Ładowanie danych testowych
X_test, y_test = mndata.load_testing()

# Przekształcenie danych w tablicę NumPy
data = np.array(X_test)

# Wyświetlanie pierwszych 5 obrazów
for i in range(5):
    # Przekształcenie wektora 784 do obrazu 28x28
    image = data[i].reshape(28, 28)
    
    # Wyświetlanie obrazu
    plt.imshow(image, cmap='gray')
    plt.title(f"Obraz {i+1}")
    plt.show()