import torch.nn as nn
from torch import cuda
from torchvision import models
from torchsummary import summary

def vgg16(params):
    # Loading the model with pretrained weights
    model = models.vgg16(pretrained=True)

    # Freezing the early layers
    for param in model.parameters():
        param.requires_grad = False
    
    n_inputs = model.classifier[6].in_features
    n_classes = 2

    # Adding to classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1)
    )

    if cuda.is_available():
        model = model.to('cuda')
    
    # print(summary(
    #     model,
    #     input_size=(3, 224, 224),
    #     batch_size=params['batch_size'],
    #     device='cuda'
    # ))

    return model