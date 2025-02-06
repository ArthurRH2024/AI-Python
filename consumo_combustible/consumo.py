import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilitar GPU para usar solo la CPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Datos de ejemplo: [peso (kg), potencia (HP), cilindros, consumo (MPG)]
datos = np.array([
    [1500, 120, 4, 25],  # Auto 1
    [1800, 150, 6, 20],  # Auto 2
    [2000, 200, 8, 15],  # Auto 3
    [1200, 80, 4, 30],   # Auto 4
    [2500, 250, 8, 12],  # Auto 5
    [1600, 130, 6, 22],  # Auto 6
    [2200, 180, 8, 18],  # Auto 7
    [1400, 100, 4, 28],  # Auto 8
], dtype=float)

# Separar características (X) y etiquetas (y)
X = datos[:, :-1]  # Características: peso, potencia, cilindros
y = datos[:, -1]   # Etiquetas: consumo (MPG)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos (escalado estándar)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, input_shape=[3], activation='relu'),  # Capa oculta 1
    tf.keras.layers.Dense(units=32, activation='relu'),  # Capa oculta 2
    tf.keras.layers.Dense(units=1)                 # Capa de salida (sin función de activación)
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),      # Optimizador Adam con tasa de aprendizaje 0.01
    loss='mean_squared_error'                      # Función de pérdida: error cuadrático medio
)

# Entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=False)
print("Modelo entrenado!")

# Evaluar el modelo en el conjunto de prueba
pérdida = modelo.evaluate(X_test, y_test, verbose=False)
print(f"Pérdida en el conjunto de prueba: {pérdida:.2f}")

# Hacer una predicción
print("\nHagamos una predicción!")
peso = float(input("Peso del automóvil (kg): "))
potencia = float(input("Potencia del motor (HP): "))
cilindros = int(input("Número de cilindros: "))

# Crear el array con los datos ingresados
nuevo_auto = np.array([[peso, potencia, cilindros]])

# Escalar los datos de entrada
nuevo_auto_escalado = scaler.transform(nuevo_auto)

# Hacer la predicción
prediccion = modelo.predict(nuevo_auto_escalado)
print(f"\nEl consumo puede ser para el automóvil es de: {prediccion[0][0]:.2f} MPG")

# Graficar la pérdida durante el entrenamiento
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.title("Pérdida durante el entrenamiento")
plt.plot(historial.history["loss"], label="Pérdida en entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida en validación")
plt.legend()
plt.show()
