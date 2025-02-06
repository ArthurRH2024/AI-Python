import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Definir transformaciones para normalizar los datos
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar dataset MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Crear DataLoaders para entrenar en lotes
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Definir la Red Neuronal
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Capa de entrada (784 -> 128)
        self.fc2 = nn.Linear(128, 64)       # Capa oculta (128 -> 64)
        self.fc3 = nn.Linear(64, 10)        # Capa de salida (64 -> 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar imagen 28x28 -> 784
        x = torch.relu(self.fc1(x))  # Activación ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No aplicamos softmax porque lo hace la función de pérdida
        return x

# Instanciar la red y definir optimizador y función de pérdida
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam

# Entrenar la red neuronal
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluar el modelo en datos de prueba
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Precisión en test: {accuracy:.2f}%")

# Visualizar algunas predicciones
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    img, label = test_dataset[i]
    with torch.no_grad():
        output = model(img.to(device).unsqueeze(0))
        pred = torch.argmax(output).item()
    
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].set_title(f"Pred: {pred}")
    axes[i].axis("off")

plt.show()
