# Predicción del valor de una casa con TensorFlow (Interactivo)

Este es un ejemplo básico de cómo entrenar una red neuronal usando TensorFlow para predecir el valor de una casa basado en sus características. Este script es interactivo y le pide al usuario que ingrese los datos de la casa para hacer la predicción.

## Tipo de problema: Regresión
Este proyecto es un ejemplo de un **problema de regresión**. En el aprendizaje automático, la regresión se utiliza para predecir valores continuos (en este caso, el precio de una casa) basado en una o más características de entrada (metros cuadrados, habitaciones, baños, garage).

### ¿Por qué se aplica regresión?
- **Valores continuos**: El precio de una casa es un valor continuo (por ejemplo, $250,000, $320,000, etc.), no una categoría o etiqueta.
- **Relación entre características y precio**: Queremos modelar la relación entre las características de la casa (como el tamaño, número de habitaciones, etc.) y su precio.

## Características utilizadas
- **Metros cuadrados**: Tamaño de la casa en metros cuadrados.
- **Número de habitaciones**: Cantidad de habitaciones en la casa.
- **Número de baños**: Cantidad de baños en la casa.
- **¿Tiene garage?**: Indica si la casa tiene garage (0 = no, 1 = sí).

## ¿Qué hace este código?
1. **Entrena una red neuronal**: Usa TensorFlow para aprender la relación entre las características de una casa y su precio.
2. **Interactúa con el usuario**: Pide al usuario que ingrese los datos de una casa para predecir su valor.
3. **Muestra una gráfica**: Grafica cómo disminuye el error (pérdida) durante el entrenamiento.

## Instrucciones para instalación y uso

### Requisitos
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib (para graficar)
- Scikit-learn (para escalado de datos)

## Posibles resultados negativos
En algunos casos, el modelo puede predecir valores negativos para el precio de una casa. Esto puede ocurrir por las siguientes razones:

- **Falta de datos de entrenamiento**: Si el conjunto de datos es pequeño o no es representativo, el modelo puede aprender patrones incorrectos.
- **Modelo no suficientemente ajustado** : Si el modelo es demasiado simple o no se entrena lo suficiente, puede no capturar la relación correcta entre las características y el precio.
- **Normalización de datos**: Si los datos no están correctamente normalizados, el modelo puede tener dificultades para hacer predicciones precisas.

## Naturaleza de la regresión:

En problemas de regresión, la salida del modelo no está restringida a valores positivos. Si el modelo aprende que ciertas combinaciones de características están asociadas con valores bajos (o negativos), puede predecir un precio negativo.

## ¿Cómo solucionarlo?
- **Aumentar los datos de entrenamiento**: Usa un conjunto de datos más grande y representativo.
- **Mejorar el modelo**: Ajusta la arquitectura de la red neuronal (más capas, más neuronas).
- **Revisar la normalización**: Asegúrate de que los datos estén correctamente normalizados.
- **Ajustar la función de pérdida**: Considera usar una función de pérdida que penalice más los errores grandes.
