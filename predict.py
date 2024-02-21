# Import necessary libraries

import argparse
import torch
import json
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Function to load the control point

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    
    # Load model architecture
    
    if arch.startswith('vgg'):
        model = models.vgg16(pretrained=False)
    elif arch.startswith('resnet'):
        model = models.resnet50(pretrained=False)
    elif arch.startswith('densenet'):
        model = models.densenet121(pretrained=False)
    else:
        print("Architecture not recognized.")
        return None
    
    # Creation of Classifier
    
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    
    # Load weights of model from checkpoint
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    return model


# Function for image processing

def process_image(image):
    img = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img)
    return img_tensor


# Function to predict the image class

def predict(image_path, model, topk=5, category_names=None, device='cpu'):
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(topk, dim=1)
    top_probabilities = top_probabilities.cpu().squeeze().tolist()
    top_indices = top_indices.cpu().squeeze().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_flowers = [cat_to_name[class_] for class_ in top_classes]
    else:
        top_flowers = top_classes
    return top_probabilities, top_classes, top_flowers


# Main function

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file to load trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes (default: 5)')
    parser.add_argument('--category_names', type=str, help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = load_checkpoint(args.checkpoint)
    probabilities, classes, flowers = predict(args.image_path, model, args.top_k, args.category_names, device)
    for probability, flower in zip(probabilities, flowers):
        print(f"{flower}: {probability:.2%}")

if __name__ == '__main__':
    main()
