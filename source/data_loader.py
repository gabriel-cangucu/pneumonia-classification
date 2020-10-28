from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_data(params):
    image_transforms = {
        # Using data augmentation on train data only
        'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                # The mean and std used are ImageNet standards
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

        'val': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        
        'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

    train_data_path = params['data_path'] + '/train'
    val_data_path = params['data_path'] + '/val'
    test_data_path = params['data_path'] + '/test'
    
    data = {
        'train': datasets.ImageFolder(root=train_data_path, transform=image_transforms['train']),
        'val': datasets.ImageFolder(root=val_data_path, transform=image_transforms['val']),
        'test': datasets.ImageFolder(root=test_data_path, transform=image_transforms['test']),
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=params['batch_size'], shuffle=True),
        'val': DataLoader(data['val'], batch_size=params['batch_size'], shuffle=True),
        'test': DataLoader(data['test'], batch_size=params['batch_size'], shuffle=True)
    }

    return data, dataloaders