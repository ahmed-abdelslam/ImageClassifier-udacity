import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# init directories
def init():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
    'trainingSets': transforms.Compose([transforms.RandomRotation(60),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]) ,
    'validationsSets': transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
    'testingSets': transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'trainingDatasets': datasets.ImageFolder(train_dir, transform = data_transforms['trainingSets']),
        'validationDatasets': datasets.ImageFolder(valid_dir, transform = data_transforms['validationsSets']),
        'testingDatasets' : datasets.ImageFolder(test_dir, transform = data_transforms['testingSets'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'trainDataloader': torch.utils.data.DataLoader(image_datasets['trainingDatasets'], batch_size = 32, shuffle = True),
        'validationDataloader': torch.utils.data.DataLoader(image_datasets['validationDatasets'], batch_size = 32, shuffle = True),
        'testingDataloader': torch.utils.data.DataLoader(image_datasets['testingDatasets'], batch_size = 32, shuffle = True)
    }
    
    return image_datasets, dataloaders

# build and train
def train(modelArchitecture, learningRate, hiddenUnits1, hiddenUnits2, epochsNO, dataloadersForTrain, dataloadersForValidate, GPUEnabled):
    if (modelArchitecture == 'vgg16'):
        model = models.vgg16(pretrained = True)
    elif (modelArchitecture == 'densenet121'):
        model = models.densenet121(pretrained = True)
    else:
        #choose by default vgg16
        print('you have choosed an incorrect architecture; the program will select vgg16 by default')
        model = models.vgg16(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hiddenUnits1)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.3)),
                              ('fc2', nn.Linear(hiddenUnits1, hiddenUnits2)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(0.2)),
                              ('fc3', nn.Linear(hiddenUnits2, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                           ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = learningRate)
    
    device = torch.device("cuda" if GPUEnabled and torch.cuda.is_available() else "cpu")
    
    model.to(device)

    epochs = epochsNO
    steps = 0
    running_loss = 0
    print_every = 50
    print("Start training ...")
    
    for epoch in range(epochs):
        
        for inputs, labels in dataloadersForTrain:
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloadersForValidate:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} || "
                      f"Train loss: {running_loss/print_every:.3f} || "
                      f"Validation loss: {test_loss/len(dataloadersForTrain):.3f} || "
                      f"Validation accuracy: {accuracy/len(dataloadersForValidate):.3f}")
                running_loss = 0
                model.train()
                        
    return model
                        
                        
def test(trainedModel, dataloadersForTest, GPUEnabled):
    totalCorrectPredctions = 0    
    dataSetSize = 0
    model = trainedModel
    print("Testing the model...")
    device = torch.device("cuda" if GPUEnabled and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloadersForTest:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, prediction = torch.max(outputs.data, 1)

            dataSetSize += labels.size(0)

            totalCorrectPredctions += (prediction == labels).sum().item()

    print(f"Accuracy : {(100 * totalCorrectPredctions / dataSetSize):.3f} %")
    
def saveCheckpoint(model, imageTrainingDatasets) :
    print("Save a checkpoint ...")
    checkpoint = {
    'model': model,
    'class_to_idx': imageTrainingDatasets.class_to_idx,
    'state_dict': model.state_dict(),
    }

    torch.save(checkpoint, 'checkpoint.pth')
    
def main():
    imageTrainingDatasets, dataloaders = init()
    print("Setup the network:")
    modelArchitecture = input("Enter name of architecture (choose between vgg16 or densenet121) >>> ")
    
    learningRate = input("Enter learning rate (recommended: 0.001) >>> ")
    learningRate = float(learningRate)
    
    hiddenUnits1 = input("Number of first hidden units (recommended: 1024) >>> ")
    hiddenUnits1 = int(float(hiddenUnits1))
    
    hiddenUnits2 = input("Number of second hidden units (recommended: 256) >>> ")
    hiddenUnits2 = int(float(hiddenUnits2))
    
    epochsNO = input("Number of epochs (recommended: 20) >>> ")
    epochsNO = int(float(epochsNO))
    
    GPUEnabled = input("Do you want to train using GPU? type 1 to use GPU, type 0 to use CPU >>> ")
    GPUEnabled = int(float(GPUEnabled))
    
    trainedModel = train(modelArchitecture, learningRate, hiddenUnits1, 
                        hiddenUnits2, epochsNO, dataloaders['trainDataloader'], dataloaders['validationDataloader'], GPUEnabled)
    test(trainedModel, dataloaders['testingDataloader'], GPUEnabled)
    saveCheckpoint(trainedModel, imageTrainingDatasets['trainingDatasets'])
    
if __name__ == '__main__': 
    main()