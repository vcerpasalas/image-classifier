import torch

def check_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    print("Keys in the checkpoint:")
    print(checkpoint.keys())

checkpoint_file = "C:\\Users\\vcerp\\Downloads\\CODIGOS\\ImageClassifier\\checkpoint.pth"
check_checkpoint(checkpoint_file)
