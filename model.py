import torch
import torchvision
from PIL import Image
from torchvision import transforms

from torch import nn


def create_effnetb2_model(num_classes:int=10, 
                          seed:int=42):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 10.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """

    # Create EffNetB2 pretrained weights, transforms and model
    
    transform = transforms.Compose([
        transforms.Resize(288, interpolation=Image.BICUBIC),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    model = torchvision.models.efficientnet_b2()

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    
    return model, transform
