import torch
import torch.nn as nn
import torchvision.models as models

# 2D Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Using a pre-trained ResNet and removing the final fully connected layer
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fc layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        return x

# Time-CNN Predictor
class TimeCNN(nn.Module):
    def __init__(self, feature_size, sequence_length, hidden_size, output_size):
        super(TimeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size * feature_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_size)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

# Assuming you have a dataset 'VideoDataset' which returns (sequence of frames, label)
train_dataset = ...  # Your training dataset
val_dataset = ...  # Your validation dataset

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Loss, and Optimizer
feature_extractor = FeatureExtractor()
predictor = TimeCNN(feature_size=2048, sequence_length=10, hidden_size=128, output_size=2048)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(predictor.parameters()), lr=0.001)

# TensorBoard
writer = SummaryWriter()

# Training Loop
def train_model(num_epochs):
    feature_extractor.train()
    predictor.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (sequence, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Process each frame through the feature extractor
            feature_sequence = []
            for t in range(sequence.shape[1] - 1):  # Exclude the last frame
                feature_sequence.append(feature_extractor(sequence[:, t]))
            feature_sequence = torch.stack(feature_sequence, dim=1)

            # Predict the future frame features
            future_features = feature_extractor(sequence[:, -1])
            predicted_features = predictor(feature_sequence)
            
            # Compute loss and backpropagate
            loss = mse_loss(predicted_features, future_features)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log to TensorBoard
        writer.add_scalar('training loss', running_loss / len(train_loader), epoch)

        # Validation Loop
        validate_model(epoch)

def validate_model(epoch):
    feature_extractor.eval()
    predictor.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, (sequence, _) in enumerate(val_loader):
            # Similar processing as in the training loop
            feature_sequence = [feature_extractor(sequence[:, t]) for t in range(sequence.shape[1] - 1)]
            feature_sequence = torch.stack(feature_sequence, dim=1)

            future_features = feature_extractor(sequence[:, -1])
            predicted_features = predictor(feature_sequence)

            loss = mse_loss(predicted_features, future_features)
            val_loss += loss.item()

        # Log validation loss to TensorBoard
        writer.add_scalar('validation loss', val_loss / len(val_loader), epoch)

train_model(num_epochs=10)
writer.close()
