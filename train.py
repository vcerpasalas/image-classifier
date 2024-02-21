# Import necessary libraries

import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
import os


# Define a function to load data

def load_data(data_dir):
    
    # Define directories for training, validation, and testing data
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transformations for the data
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets and create data loaders
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'valid', 'test']}
    
    return dataloaders['train'], dataloaders['valid'], image_datasets['train'].class_to_idx


# Define a function to build the model architecture

def build_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model


# Function to train the model

def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, device='cuda'):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        valid_loss /= len(valid_loader.dataset)
        accuracy /= len(valid_loader)
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Validation loss: {valid_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}")


# Function to save the checkpoint

def save_checkpoint(model, train_data, save_dir, arch, epochs, hidden_units):
    checkpoint = {
        'arch': arch,
        'epochs': epochs,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier 
    }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))


# Main function

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Choose architecture (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Set learning rate (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set number of hidden units (default: 512)')
    parser.add_argument('--epochs', type=int, default=20, help='Set number of epochs (default: 20)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    train_model(model, criterion, optimizer, train_loader, valid_loader, args.epochs, device)
    save_checkpoint(model, train_loader.dataset, args.save_dir, args.arch, args.epochs, args.hidden_units)

if __name__ == '__main__':
    main()
