import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from collections import OrderedDict
import json
import numpy as np
import os




def get_input_args():
    parser = argparse.ArgumentParser(description = 'Flower Classifier')
    parser.add_argument('--data_dir', type = str, default="./flowers",
                                 help = 'dataset path')
    parser.add_argument('--arch',type= str, default = 'vgg16',
                                 help = 'vgg16 model')
    parser.add_argument('--input_size', type = int, default = 25088,
                                 help='Number of Inputs')
    parser.add_argument('--hidden_one', type = int, default = 4096,
                                 help='Number of hidden units in layer 1')
    parser.add_argument('--dropout', type = int, default = 0.5,
                                 help='dropout')
    parser.add_argument('--learning_rate', type = int, default = 0.0001,
                                 help='learning rate of the optimiser')
    parser.add_argument('--gpu', action = "store", default = 'cuda',
                                 help = 'either CPU or GPU if available')
    parser.add_argument('--epochs', type = int, default = 1,
                                 help='Number of Epochs')
    parser.add_argument('--save_dir', type = str, default='/home/workspace/ImageClassifier/vgg16',
                                 help = ' path to checkpoint file')
    args = parser.parse_args()
    return args


def transform(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    all_transforms = {'train_transforms' : transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
                      
                      'test_transforms': transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
                      
                     'validation_transforms' : transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])}
    
    
    all_datasets = {'train_datasets' : datasets.ImageFolder(train_dir, transform=all_transforms['train_transforms']),
                    'test_datasets' : datasets.ImageFolder(test_dir, transform=all_transforms['test_transforms']),
                    'validation_datasets' : datasets.ImageFolder(valid_dir, transform=all_transforms['validation_transforms'])}
 
    all_loaders ={'trainingloader' : torch.utils.data.DataLoader(all_datasets['train_datasets'], batch_size=50, shuffle=True),
                  'testloader' : torch.utils.data.DataLoader(all_datasets['test_datasets'], batch_size=50, shuffle=True),
                  'validationloader' : torch.utils.data.DataLoader(all_datasets['validation_datasets'], batch_size=256, shuffle=True)}
    
    return all_transforms, all_datasets, all_loaders


def network_classifier(arch,input_size, hidden_one, dropout, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch == 'vgg16'
    model = models.vgg16(pretrained=True)
       
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(input_size, hidden_one),
                           nn.ReLU(),
                           nn.Dropout(dropout),
                           nn.Linear(hidden_one,102),
                           nn.LogSoftmax(dim = 1))

    model.classifier = classifier
    #Define loss
    criterion = nn.NLLLoss()
    #Define optimizer with classifier and learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)
    
    return model, criterion, optimizer

def training(device, epochs, model, all_loaders, optimizer, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 0
    steps = 0
    running_loss = 0
    print_every =20
    for epoch in range(epochs):
        for images, labels in all_loaders['trainingloader']:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
       
            
            with torch.no_grad():
                for images , labels in all_loaders['validationloader']:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps,labels)
                    test_loss += loss.item()
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.cuda.FloatTensor)).item()
            print("Epoch: {}/{}.." .format(epoch+1, epochs),
                  "Train loss: {:.3f}..".format(running_loss/print_every),
                  "Validation loss: {:.3f}.." .format(test_loss/len(all_loaders['validationloader'])),
                  "Validation accuracy: {:.2f}%".format(accuracy/len(all_loaders['validationloader'])*100))
            running_loss = 0
            model.train()  
    return model  
print("Training successful")


def test(device, all_loaders, model, criterion):
    epochs = 0
    steps = 0
    running_loss = 0
    print_every =20
    for epoch in range(epochs):
        for images, labels in all_loaders['trainingloader']:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
       
            
            with torch.no_grad():
                for images , labels in all_loaders['testloader']:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps,labels)
                    test_loss += loss.item()
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.cuda.FloatTensor)).item()
            print("Epoch: {}/{}.." .format(epoch+1, epochs),
                  "Train loss: {:.3f}..".format(running_loss/print_every),
                  "Test loss: {:.3f}.." .format(test_loss/len(all_loaders['testloader'])),
                  "Test accuracy: {:.2f}%".format(accuracy/len(all_loaders['testloader'])*100))
            running_loss = 0
            model.train()
    return None
print("Testing successful")
    
def saving_checkpoint(arch, model, input_size, epochs, all_datasets, optimizer):
    model.class_to_idx = all_datasets['train_datasets'].class_to_idx
    checkpoint = {'arch':arch,
                  'classifier': model.classifier,
                  'input_size': input_size ,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'epochs' :epochs,
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    
    torch.load(os.path.join(checkpoint, "checkpoint.pth"), map_location=lambda storage, loc: storage)
    #torch.save(checkpoint, save_dir + '/checkpoint.pth')
    #torch.save(checkpoint, 'checkpoint.pth')
    return None
print("Saving Checkpoint function carried out")

def main():
    in_args = get_input_args()
    arch = in_args.arch 
    data_dir = in_args.data_dir
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'
    device = in_args.gpu
    input_size = in_args.input_size
    hidden_one = in_args.hidden_one
    dropout = in_args.dropout
    lr = in_args.learning_rate
    epochs = in_args.epochs
    checkpoint = in_args.save_dir
    
    all_transforms , all_datasets, all_loaders = transform(data_dir)
    model, criterion, optimizer= network_classifier(arch,input_size, hidden_one, dropout, lr)
    training(device, epochs, model, all_loaders, optimizer, criterion)
    test(device, all_loaders, model, criterion)
    saving_checkpoint(arch, model, input_size, epochs, all_datasets, optimizer)
    
if __name__ == "__main__":
    main()