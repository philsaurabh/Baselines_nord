import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


# BiFPN Block Definition
class BiFPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        return x


# Teacher Network with BiFPN
class TeacherNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TeacherNetwork, self).__init__()
        backbone = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Exclude avgpool and fc layers
        self.bifpn = nn.Sequential(
            BiFPNBlock(512, 256),
            BiFPNBlock(256, 128)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(128, num_classes)  # Final fully connected layer

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, 512, H, W]
        features = self.bifpn(features)  # BiFPN processing
        pooled_features = self.global_pool(features).view(features.size(0), -1)  # Flatten
        output = self.fc(pooled_features)  # Classification
        return output


# Student Network
class StudentNetwork(nn.Module):
    def __init__(self, num_classes):
        super(StudentNetwork, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        output = self.fc(features)
        return output


# Apply Symmetric Label Noise
def apply_symmetric_noise(labels, num_classes, noise_rate):
    labels = np.array(labels)
    n = len(labels)
    noisy_labels = labels.copy()
    num_noisy = int(n * noise_rate)
    noisy_indices = np.random.choice(n, num_noisy, replace=False)
    for idx in noisy_indices:
        current_label = labels[idx]
        noisy_label = np.random.choice([l for l in range(num_classes) if l != current_label])
        noisy_labels[idx] = noisy_label
    return torch.tensor(noisy_labels, dtype=torch.long)


# Metrics Calculation
def calculate_metrics(predictions, targets):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    accuracy = (predictions == targets).mean() * 100
    f1 = f1_score(targets, predictions, average="weighted")
    mcc = matthews_corrcoef(targets, predictions)
    return accuracy, f1, mcc
# Train a Model
def train_model(model, data_loader, criterion, optimizer, epochs, device, noisy_labels=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy, total_f1, total_mcc = 0.0, 0.0, 0.0
        num_batches = len(data_loader)

        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = noisy_labels[i * data_loader.batch_size:(i + 1) * data_loader.batch_size].to(device) \
                if noisy_labels is not None else labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            accuracy, f1, mcc = calculate_metrics(predictions, labels)
            total_accuracy += accuracy
            total_f1 += f1
           # total_mcc += mcc

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}, "
              f"Accuracy: {total_accuracy / num_batches:.2f}%, "
              f"F1-Score: {total_f1 / num_batches:.4f} ")
             # f"MCC: {total_mcc / num_batches:.4f}")

# Test Model
def test_model(model, test_loader, device):
    model.eval()
    total_accuracy, total_f1, total_mcc = 0.0, 0.0, 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            accuracy, f1, mcc = calculate_metrics(predictions, labels)
            total_accuracy += accuracy
            total_f1 += f1
           # total_mcc += mcc

    print(f"Test Results -> Accuracy: {total_accuracy / num_batches:.2f}%, "
          f"F1-Score: {total_f1 / num_batches:.4f}")#, MCC: {total_mcc / num_batches:.4f}")


# Distill Knowledge to Student
def distill_student(student_model, teacher_model, data_loader, optimizer, epochs, device, noisy_labels=None):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()

    student_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_accuracy, total_f1, total_mcc = 0.0, 0.0, 0.0
        num_batches = len(data_loader)
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            labels = noisy_labels[i * data_loader.batch_size:(i + 1) * data_loader.batch_size].to(device) \
                if noisy_labels is not None else labels.to(device)
            # Get teacher predictions (soft labels)
            teacher_model.eval()
            with torch.no_grad():
                soft_labels = torch.softmax(teacher_model(images), dim=1)

            # Student predictions
            student_outputs = student_model(images)

            # Combined loss
            loss = 0.7 * kl_loss(torch.log_softmax(student_outputs, dim=1), soft_labels) + \
                   0.3 * ce_loss(student_outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = student_outputs.argmax(dim=1)
            accuracy, f1, mcc = calculate_metrics(predictions, labels)
            total_accuracy += accuracy
            total_f1 += f1
            total_mcc += mcc

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}, "
              f"Accuracy: {total_accuracy / num_batches:.2f}%, "
              f"F1-Score: {total_f1 / num_batches:.4f}, ")
              #f"MCC: {total_mcc / num_batches:.4f}")


# Main Execution
if __name__ == "__main__":
    # Device Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Preparation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])

    # Train and Test Datasets
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Apply Symmetric Label Noise
    noisy_labels = apply_symmetric_noise(train_dataset.targets, num_classes=100, noise_rate=0.6).to(device)

    # Train Teacher Model
    teacher_model = TeacherNetwork(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    print("Training Teacher Model with Noisy Labels...")
    train_model(teacher_model, train_loader, criterion, optimizer, epochs=0, device=device, noisy_labels=noisy_labels)

    # Test Teacher Model
    print("Testing Teacher Model...")
    test_model(teacher_model, test_loader, device)

    # Distill Knowledge to Student Model
    student_model = TeacherNetwork(num_classes=100).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    print("Distilling Knowledge to Student Model...")
    distill_student(student_model, teacher_model, train_loader, optimizer, epochs=1, device=device, noisy_labels=noisy_labels)

    # Test Student Model
    print("Testing Student Model...")
    test_model(student_model, test_loader, device)
