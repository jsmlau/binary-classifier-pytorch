import torch
import torch.nn as nn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import shutil
import time
import copy
from PIL import Image
import glob

weight_file_path = './saved_weights/A_ep-7_loss-0.1851.pth'
model = torch.load(weight_file_path)

def process_image(image):

    pil_image = Image.open(image)
   
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img




def classify(image_path):
    device = torch.device("cpu")
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()

    model.eval()
    model.cpu()
    output = model(img)

    _, predicted = torch.max(output, 1)

    output = predicted.data[0].cpu().detach().numpy()
    print(output)
    return output

if __name__ == '__main__':
    map_location=torch.device('cpu')
    out_path = './output'
    labels = ['face', 'noface']
    for l in labels:
        if not os.path.isdir(os.path.join(out_path,l)):
            os.makedirs(os.path.join(out_path,l))    

    for image_path in glob.glob('./to_be_labeled/*png'):
        label = classify(image_path)
        if label == 0:
            shutil.copy(image_path, os.path.join(out_path,labels[0]))
        else:
            shutil.copy(image_path, os.path.join(out_path,labels[1]))








