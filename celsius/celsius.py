# Ejercicio Universal de Entrenamiento de IA con un ejemplo de conversion de Celsius a Fahrenheit (Regresion)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                  # Deshabilitar GPU para usar solo la CPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

# Datos de entrenamiento: pares de valores Celsius y Fahrenheit
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definir las capas de la red neuronal
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])  # Capa oculta 1 con 3 neuronas
oculta2 = tf.keras.layers.Dense(units=3)                   # Capa oculta 2 con 3 neuronas
salida = tf.keras.layers.Dense(units=1)                    # Capa de salida con 1 neurona

# Crear el modelo secuencial con las capas definidas
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),               # Usar el optimizador Adam con una tasa de aprendizaje de 0.1
    loss='mean_squared_error'                              # Función de pérdida: error cuadrático medio
)

# Entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)  # Entrenar por 1000 épocas
print("Modelo entrenado!")

# Hacer una predicción
print("Hagamos una predicción!")
resultado = modelo.predict(np.array([100.0]).reshape(1, 1))  # Predecir la conversión de 100°C a Fahrenheit
print(f"El resultado es {resultado[0][0]:.2f}°F")            # Formatear la salida de la predicción

# Mostrar los pesos (parámetros) aprendidos por el modelo
print("Variables internas del modelo")
print("Pesos y sesgos de la capa oculta 1:")
print(oculta1.get_weights())                                 # Pesos y sesgos de la primera capa oculta
print("Pesos y sesgos de la capa oculta 2:")
print(oculta2.get_weights())                                 # Pesos y sesgos de la segunda capa oculta
print("Pesos y sesgos de la capa de salida:")
print(salida.get_weights())                                  # Pesos y sesgos de la capa de salida

# Graficar la pérdida durante el entrenamiento
plt.xlabel("# Época")                                      # Etiqueta del eje X
plt.ylabel("Magnitud de pérdida")                          # Etiqueta del eje Y
plt.title("Pérdida durante el entrenamiento")              # Título de la gráfica
plt.plot(historial.history["loss"])                        # Graficar la pérdida
plt.savefig("grafica_perdida.png")                         # Guardar la gráfica en un archivo
plt.show()                                                 # Mostrar la gráfica
