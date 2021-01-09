import json
import torch
import numpy as np
from torch import optim
from torchvision import models
from PIL import Image

def loadCheckpoint(checkpoint_path) :
    checkpoint = torch.load(checkpoint_path)
    
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # resize the image
    resizeTo = 256,256
    image.thumbnail(resizeTo)

    # get size of the image in pixels
    width, height = image.size
    
    # setting the points for cropped image
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # normalize the image
    means = np.array([0.485, 0.456, 0.406])
    standardDeviations = np.array([0.229, 0.224, 0.225])
    imageArray = np.array(image) / 255
    image = (imageArray - means) / standardDeviations
    
    # transpose the image
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)

def predict(imageToTensor, model, topk, cat_to_name, GPUEnabled):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = imageToTensor
    image = image.unsqueeze_(0)
    image = image.cuda().float()
    
    model.eval()
    
    device = torch.device("cuda" if GPUEnabled and torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        probabilities, idxs = torch.topk(output, topk)
        idxs = np.array(idxs)            
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]
        names = []
        for Class in classes:
            names.append(cat_to_name[str(Class)])
        
        probabilities = [100 * np.exp(probability) for probability in probabilities[0]]

        return zip(probabilities, names)
    
def printProbability(predictions):
    for probability, name in predictions:
        print(f'The probability of {name}: {probability:.3f} %')

def main():
    imageToPredict = input('Enter the path of image >>> flowers/test/')
    checkpoint = input('Enter the path of checkpoint >>> ')
    jsonFile = input('Enter the path of json file >>> ')
    topk = input('Enter topk >>> ')
    topk = int(float(topk))
    GPUEnabled = input("Do you want to train using GPU? type 1 to use GPU, type 0 to use CPU >>> ")
    GPUEnabled = int(float(GPUEnabled))
    
    with open(jsonFile, 'r') as f:
        cat_to_name = json.load(f)
    model = loadCheckpoint(checkpoint)
    imageToTensor = process_image(Image.open('flowers/test/' + imageToPredict))
    predictions = predict(imageToTensor, model, topk, cat_to_name, GPUEnabled)
    printProbability(predictions)
    
if __name__ == '__main__': 
    main()