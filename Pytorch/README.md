# Red Neuronal para Clasificación de Dígitos (MNIST) con PyTorch

Este proyecto entrena una **red neuronal totalmente conectada** para clasificar imágenes del dataset **MNIST**, que contiene dígitos escritos a mano (0-9). Utiliza **PyTorch** para construir y entrenar el modelo.

## ¿Qué hace este código?
1- Este código en Python entrena una red neuronal para reconocer dígitos escritos a mano (0-9) usando el dataset MNIST.

2- Convierte las imágenes a tensores y las normaliza para mejorar el aprendizaje.

3- Descarga el dataset MNIST (imágenes de 28x28 píxeles).

4- Crea DataLoaders para cargar las imágenes en lotes de 64 (para entrenar más rápido)

## Crea la Red Neuronal
Tiene tres capas:

     - Entrada: 784 neuronas (porque la imagen de 28x28 se aplana).
     
     - Ocultas: 128 y 64 neuronas con activación ReLU.
     
     - Salida: 10 neuronas (una por cada número del 0 al 9).
     

## Muestra Predicciones
Toma 5 imágenes de prueba.
Predice el número y las muestra en pantalla.
📌 Salida esperada: Una precisión de ~97% y una imagen con ejemplos de predicción. 🎯

## Instrucciones para instalación y uso

### Requisitos
- Python 3.x
  
- torch
  
- torchvision
  
- Matplotlib (para graficar)


Si tienes más preguntas o necesitas ayuda, ¡no dudes en preguntar! 😊
