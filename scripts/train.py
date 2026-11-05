import argparse
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

DEFAULT_TRAIN_PATH = Path('data_reduced/train')
DEFAULT_MODEL_PYTORCH_PATH = Path('models/orange_diseases_model_1.pth')
DEFAULT_MODEL_ONNX_PATH = Path('models/orange_diseases_model_1.onnx')
LEARNING_RATE = 0.01
INNER_LAYER_SIZE = 10
DROPOUT_RATE = 0.2
EPOCHS = 20
INPUT_SIZE = 224
INTERPOLATION = transforms.InterpolationMode.NEAREST
ROTATION_DEGREES = 10
CROP_SCALE = (0.9, 1.0)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 0

def get_classes(train_path) -> list[str]:
    if not os.path.exists(train_path):
        print(f"Error: folder not found {train_path}")
        return []
    classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    return classes

class Model(nn.Module):
    def __init__(self, size_inner, droprate, num_classes):
        super().__init__()
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Remove original classifier
        self.base_model.classifier = nn.Identity()
        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)  # New inner layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)  # Add dropout
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x

def make_model(learning_rate: float, size_inner: int, droprate: float, device: torch.device, num_classes: int) -> tuple[Model, optim.Adam]:
    model = Model(size_inner=size_inner, droprate=droprate, num_classes=num_classes)
    model.to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    return model, optimizer

def get_train_loader(
        train_path: Path, 
        input_size: int, 
        crop_scale: tuple[float, float], 
        mean: list[float], 
        std: list[float], 
        num_workers: int) -> DataLoader:
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=INTERPOLATION),
        transforms.RandomRotation(ROTATION_DEGREES), 
        transforms.RandomResizedCrop(input_size, crop_scale), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_dataset = ImageFolder(train_path, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader

def train_and_save(model, optimizer, train_loader, criterion, num_epochs, device, save_path):
    print(f"Training started for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved in: {save_path}")

def save_onnx(model: Model, modelpth_path: Path, modelonnx_path: Path):
    model.load_state_dict(torch.load(modelpth_path, weights_only=True, map_location='cpu'))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Asegurar que la carpeta destino de ONNX existe
    dir_name = os.path.dirname(modelonnx_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    torch.onnx.export(
        model,
        dummy_input,
        modelonnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"\n✅ Model exported to ONNX format in: {modelonnx_path}")

def process_training(train_path: Path, modelpth_path: Path, modelonnx_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Cambiado a dinámico por si tienes GPU disponible
    criterion = nn.CrossEntropyLoss()
    
    classes = get_classes(train_path)
    if not classes:
        print("❌ Aborting training: No classes found in training directory.")
        return
        
    model, optimizer = make_model(LEARNING_RATE, INNER_LAYER_SIZE, DROPOUT_RATE, device, len(classes))
    train_loader = get_train_loader(train_path, INPUT_SIZE, CROP_SCALE, MEAN, STD, NUM_WORKERS)
    train_and_save(model, optimizer, train_loader, criterion, EPOCHS, device, modelpth_path)
    save_onnx(model, modelpth_path, modelonnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Network Model")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--modelpth", type=Path, default=DEFAULT_MODEL_PYTORCH_PATH)
    parser.add_argument("--modelonnx", type=Path, default=DEFAULT_MODEL_ONNX_PATH)
    args = parser.parse_args()
    process_training(args.train, args.modelpth, args.modelonnx)
