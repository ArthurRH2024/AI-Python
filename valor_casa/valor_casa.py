import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                              # Deshabilitar GPU para usar solo la CPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Datos de ejemplo: [metros_cuadrados, habitaciones, baños, garage (0 = no, 1 = sí)]
caracteristicas = np.array([
    [120, 3, 2, 1],  # Casa 1
    [150, 4, 2, 0],  # Casa 2
    [80, 2, 1, 0],   # Casa 3
    [200, 5, 3, 1],  # Casa 4
    [100, 3, 2, 1],  # Casa 5
], dtype=float)

# Valores de las casas (en miles de dólares)
precios = np.array([250, 320, 180, 450, 220], dtype=float)

# Escalado estándar de los datos
scaler = StandardScaler()
caracteristicas = scaler.fit_transform(caracteristicas)
precios = precios / 1000                                                  # Normalizar precios (dividir entre 1000)

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, input_shape=[4], activation='relu'),  # Capa oculta 1
    tf.keras.layers.Dense(units=8, activation='relu'),                    # Capa oculta 2
    tf.keras.layers.Dense(units=1)                                        # Capa de salida (sin función de activación)
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),                             # Tasa de aprendizaje más baja
    loss='mean_squared_error'                                             # Función de pérdida: error cuadrático medio
)

# Entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(caracteristicas, precios, epochs=1000, verbose=False) # Entrenar con 1000 vueltas o Epocas
print("Modelo entrenado!")

# Pedir datos al usuario para hacer una predicción
print("\nIngresa los datos de la casa para predecir su valor:")
metros_cuadrados = float(input("Metros cuadrados: "))
habitaciones = int(input("Número de habitaciones: "))
banos = int(input("Número de baños: "))
garage = int(input("¿Tiene garage? (0 = no, 1 = sí): "))

# Crear el array con los datos ingresados
nueva_casa = np.array([[metros_cuadrados, habitaciones, banos, garage]], dtype=float)

# Escalar los datos de la nueva casa
nueva_casa_escalada = scaler.transform(nueva_casa)

# Hacer la predicción
prediccion = modelo.predict(nueva_casa_escalada)
print(f"\nEl valor predicho para la casa es: ${prediccion[0][0] * 1000:.2f} miles de dólares")

# Graficar la pérdida durante el entrenamiento
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.title("Pérdida durante el entrenamiento")
plt.plot(historial.history["loss"])
plt.show()
