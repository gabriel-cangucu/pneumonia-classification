import numpy as np
import argparse
import os
from datetime import datetime
from timeit import default_timer as timer

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Importing custom models
from models import vgg16


def get_params(args):
    params = {}

    params['data_path'] = args.data_path
    params['results_path'] = args.results_path
    params['batch_size'] = args.batch_size
    params['optimizer'] = args.optimizer
    params['criterion'] = args.criterion
    params['learning_rate'] = args.learning_rate
    params['num_epochs'] = args.num_epochs
    params['execution_start_time'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    return params


def get_model_name(params):
    model_name = f'vgg16_batch{params["batch_size"]}_op{params["optimizer"]}_loss{params["criterion"]}_lr{params["learning_rate"]}_epochs{params["num_epochs"]}_{params["execution_start_time"]}'
    return model_name

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


def evaluate_model(model, criterion, data_loader, train_on_gpu=True):
    val_loss = 0.0
    val_acc = 0

    # No need to keep track of gradients
    with torch.no_grad():
        model.eval()

        for features, labels in data_loader['val']:
            # Tensor to gpu if available
            if train_on_gpu:
                features, labels = features.cuda(), labels.cuda()

            # Forward propagation
            outputs = model(features)

            # Calculating validation loss
            loss = criterion(outputs, labels)

            # Multiplying average loss by the number of examples in batch
            val_loss += loss.item() * features.size(0)

            # Calculating validation accuracy
            _, pred = torch.max(outputs, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            # Multiplying average accuracy by the number of examples
            val_acc += accuracy.item() * features.size(0)
        
        # Calculating average loss
        val_loss = val_loss / len(data_loader['val'].dataset)
        # Calculating average accuracy
        val_acc = val_acc / len(data_loader['val'].dataset)
    
    return val_loss, val_acc

def train_model(params, data, data_loader, results_file, train_on_gpu=True):
    model = vgg16(params)

    # Mapping class names to indexes
    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    # Setting the loss function
    if params['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif params['criterion'] == 'log_likelihood':
        criterion = nn.NLLLoss()
    else:
        raise ValueError("Criterion not found. Options available are 'cross_entropy' and 'log_likelihood'.")

    # Setting the optimizer
    if params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    else:
        raise ValueError("Optimizer not found. Options available are 'sgd' and 'adam'.")

    print(f'\nBegan training.\n')
    start = timer()

    train_acc_list = []
    train_loss_list = []

    val_acc_list = []
    val_loss_list = []

    # Keeping track of the best model
    best_val_loss = np.inf
    best_val_acc = np.inf

    for epoch in range(1, params['num_epochs']+1):
        train_loss = 0.0
        train_acc = 0

        # Training the model and keeping track of the time
        model.train()
        epoch_start = timer()

        for i, data in enumerate(data_loader['train']):
            features, labels = data

            # Tensor to gpu if available
            if train_on_gpu:
                features, labels = features.cuda(). labels.cuda()

            # Setting parameter gradients to zero
            optimizer.zero_grad()

            # Forward propagataion
            outputs = model(features)

            # Backward propagation
            loss = criterion(outputs, labels)
            loss.backward()

            # Updating the parameters
            optimizer.step()
            # Train loss is multiplied by number of examples in batch
            train_loss += loss.item() * features.size(0)

            # Calculating accuracy by finding max log probability
            _, pred = torch.max(outputs, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))

            if train_on_gpu:
                correct = np.squeeze(correct_tensor.cpu().numpy())
            else:
                correct = np.squeeze(correct_tensor.numpy())
            
            # Converting int to float
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Accuracy is multiplied by the number of examples in batch
            train_acc += accuracy.item() * features.size(0)

            # Printing intermediate results
            time = timer() - start
            print(f'Epoch: {epoch} \t {100 * (i + 1) / len(data_loader["train"]):.2f}% complete. {time:.2f} seconds elapsed in epoch.', end='\r')
        
        # Calculating average losses
        train_loss = train_loss / len(data_loader['train'].dataset)
        # Calculating average accuracy
        train_acc = train_acc / len(data_loader['train'].dataset)

        # Begin the validation process
        val_loss, val_acc = evaluate_model(model, criterion, data_loader)

        # If the loss decreased
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), results_file)

            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch

        # Saving all results
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f'\nEpoch: {epoch} \t Training Loss: {train_loss:.4f}', end='\r')
        print(f'\tTraining Accuracy: {100 * train_acc:.2f}%')

        print(f'\nEpoch: {epoch} \t Validation Loss: {val_loss:.4f}', end='\r')
        print(f'\tValidation Accuracy: {100 * val_acc:.2f}%')
    
    # Calculating the total time elapsed and printing results
    total_time = timer() - start
    print(f'\nDone training! Total time elapsed: {total_time:.2f} seconds.')
    print(f'Best epoch: {best_epoch} with loss {best_val_loss:.2f} and accuracy {100 * best_val_acc}.%')
    print(f'Results saved in {results_file}.')


def main(args):
    # Getting the parser arguments
    params = get_params(args)

    # Checking if a gpu is available
    # cuda.set_device(0)
    train_on_gpu = cuda.is_available()
    print(f'Training on GPU: {train_on_gpu}\n')

    # Loading the data loader and applying transforms 
    data, data_loader = load_data(params)

    # Analysing data splits
    splits = ['train', 'val', 'test']

    for split in splits:
        split_iter = iter(data_loader[split])
        features, _ = next(split_iter)

        print(f'\'{split}\' split has {features.shape[0]} instances.')

    model_name = get_model_name(params)
    train_model(params, data, data_loader, model_name, train_on_gpu=train_on_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a model capable of identifying pneumonia in chest X Ray images.')

    # Default paths when running the code from the source folder
    default_data_path = os.path.abspath(os.path.join(os.path.abspath(''), '../data'))
    default_results_path = os.path.abspath(os.path.join(os.path.abspath(''), '../results'))

    parser.add_argument('-dp', '--data-path', type=str, default=default_data_path, dest='data_path',
                        help='Path to the dataset.')
    parser.add_argument('-rp', '--results-path', type=str, default=default_results_path, dest='results_path',
                        help='Path to the folder where execution results are saved.')
    parser.add_argument('-bs', '--batch-size', type=int, default=128, dest='batch_size',
                        help='Size of the batch to be feed to the network. Default is 128.')
    parser.add_argument('-op', '--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', dest='optimizer',
                        help='Name of the optimizer to be used. Default is SGD.')
    parser.add_argument('-cr', '--criterion', type=str, choices=['cross_entropy', 'log_likelihood'], default='cross_entropy', dest='criterion',
                        help='Criterion used for the loss function. Default is Cross Entropy.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, dest='learning_rate',
                        help='Learning rate used by the optimizer. Default is 0.001.')
    parser.add_argument('-ep', '--num-epochs', type=int, default=10, dest='num_epochs',
                        help='Number of epochs to train the machine learning model. Default is 10.')
    
    main(parser.parse_args())