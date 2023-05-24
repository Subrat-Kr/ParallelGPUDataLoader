import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from skimage import io, img_as_ubyte, exposure
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torchvision.models as models
import wandb

# Start a new W&B run
# wandb.init(project='BACH_gnet_0.0', entity='subratkr')

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class BACH(Dataset):
    def __init__(self, csv_file=None, root_dir=None, transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = 'bach'
        self.train_labels = self.csv_data["label"]
        self.nb_classes = 4

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        label = self.csv_data.loc[index, 'label']
        img_path = self.csv_data.loc[index, 'Name']
        img_path = os.path.join(self.root_dir, img_path)

        image = io.imread(img_path)
        image = img_as_ubyte(exposure.rescale_intensity(image))

        if self.transform:
            image = self.transform(image)

        return (image, label, index)

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Set the paths and filenames
train_root_dir = '/home/Drive4/drive1/subrat/subrat_19may23_final/Data/train_image/'
train_csv_file = '/home/Drive4/drive1/subrat/subrat_19may23_final/Final_Supervised_Results/noisy_bach_pf.csv'

# Create the BACH dataset instance
dataset = BACH(csv_file=train_csv_file, root_dir=train_root_dir, transform=transform)

# Define the number of folds for cross-validation
num_folds = 5

# Create the K-fold splitter
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

# Initialize lists to store results
accuracies = []
losses = []

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Perform K-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(kf.split(dataset), 1):
    print(f"Fold {fold}")

    # Create the data loaders for training and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=10, shuffle=False)

    # Define your model and other necessary components (GoogLeNet)
    model = models.googlenet(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, dataset.nb_classes)

    # Utilize DataParallel to use multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Train the model for each epoch
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")

        # Train the model
        model.train()
        T_correct = 0
        T_total = 0
        running_loss = 0.0

        for images, labels, _ in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculating Training Accuracy
            _, predicted = torch.max(outputs.data, 1)
            T_total += labels.size(0)
            T_correct += (predicted == labels).sum().item()

        Training_acc = 100 * T_correct / T_total
        epoch_loss = running_loss / len(train_loader)
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"Training acc: {Training_acc}")
        # wandb.log({'Train_loss': epoch_loss, 'Train_acc': Training_acc}, step=num_epochs)

        # Evaluate the model on validation data
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation acc: {val_accuracy:.4f}")
        # wandb.log({'Val_loss': val_loss, 'Val_acc': val_accuracy}, step=num_epochs)
        
        # wandb.finish()

        # Store the results
        accuracies.append(val_accuracy)
        losses.append(val_loss)

# Calculate and print the average results over all folds
avg_accuracy = torch.tensor(np.mean(accuracies), device=device)
avg_loss = torch.tensor(np.mean(losses), device=device)
print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
print(f"Average Loss: {avg_loss:.4f}")

