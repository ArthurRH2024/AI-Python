import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar el dataset MNIST (70,000 imágenes de dígitos 0-9)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int64)  # Convertir etiquetas a enteros

# Dividir en datos de entrenamiento (60,000) y prueba (10,000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, random_state=42)

# Normalizar los datos (Mejora el rendimiento del modelo)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir la Red Neuronal con MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=20, random_state=42)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Evaluar el modelo
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión en test: {accuracy:.2%}")

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", conf_matrix)

# Visualizar algunas predicciones
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    img = X_test[i].reshape(28, 28)  # Convertir vector de 784 a imagen 28x28
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Pred: {y_pred[i]}")
    axes[i].axis("off")

plt.show()
