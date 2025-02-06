# Red Neuronal (MLPClassifier) de Scikit-Learn para reconocer dígitos escritos a mano.

 Este es un ejemplo usando Scikit-Learn para entrenar un modelo de clasificación en el dataset MNIST.

## ¿Qué hace este código?
- Cargar el Dataset MNIST. Contiene 70,000 imágenes de dígitos (28x28 píxeles aplanados en 784 valores). 
- Dividir los Datos. 60,000 imágenes para entrenamiento y 10,000 para prueba.
- Normalizar los Datos. Se usa StandardScaler() para escalar los valores entre -1 y 1 (ayuda a la red a aprender mejor).
- Definir el Modelo (MLPClassifier).

## Crea la Red con 2 capas ocultas:
-128 neuronas en la primera capa.

-64 neuronas en la segunda.

-Activación ReLU, optimizador Adam y 20 iteraciones.

-Entrenar el Modelo. Aprende a reconocer los números a partir de las imágenes de entrenamiento.

-Evaluar el Modelo. Predice sobre imágenes de prueba y calcula la precisión.

-Matriz de Confusión. Muestra cómo se confunden los dígitos entre sí.

-Visualizar Resultados. Muestra 5 imágenes de prueba junto con sus predicciones.

    

## Muestra Predicciones
📌 Salida esperada: Una precisión de ~97% y una imagen con 5 ejemplos de predicción. 🚀

## Instrucciones para instalación y uso

### Requisitos
- Python 3.x
  
- 
  
- torchvision
