
import os, shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys

import cv2
import time
import json
from datetime import datetime

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models 
import torch.nn.functional as F

import numpy as np

def trainer(ops):
    try:
        if ops.num_classes == 1:
            sigmoid_output = True
        elif ops.num_classes == 2:
            sigmoid_output = False
        
        # Use pretrained weights from torch.utils.model_zoo
        model_ = models.resnet18(pretrained=True)
        for (name, module) in model_.named_children():
            print(name)
        num_features = model_.fc.in_features
        if sigmoid_output:
            model_.fc = torch.nn.Sequential( 
                nn.Linear(num_features, ops.num_classes),
                nn.Sigmoid()
            )
        else:
            model_.fc = nn.Linear(num_features, ops.num_classes)

        # freeze all layers except fc
        #for param in model_.parameters():
        #    param.requires_grad = False
        #for param in model_.fc.parameters():
        #    param.requires_grad = True  
        #for name, param in model_.named_parameters():
        #    print('Name: ', name, 'Requires_Grad: ', param.requires_grad)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)

        # ./train/face/img_1.png
        # ...
        # ./train/noface/img_1.png
        # ...
        # ./val/face/img_1.png
        # ...
        # ./val/noface/img_1.png
        # ...
        train_dir = './train/'
        val_dir = './val/'

        train_loader = data.DataLoader(
            datasets.ImageFolder(train_dir,
                                transforms.Compose([
                                    transforms.RandomAffine(degrees=(-90, 90), translate=(0.1, 0.3), scale=(1.0, 1.3)),
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(scale=(0.02, 0.06), ratio=(0.8, 1.2),value=(255,255,255)),
                                    # imagenet mean ana std
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])),
            batch_size=ops.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True)


        val_loader = data.DataLoader(
            datasets.ImageFolder(val_dir,
                                transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])),
            batch_size=ops.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True)      

        print(f'Num training images: {len(train_loader.dataset)}')
        print(f'Num validation images: {len(val_loader.dataset)}')
        
        optimizer = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum)

        # for my lr_scheduler 
        best_loss = np.inf
        flag_change_lr_cnt = 0 
        init_lr = ops.init_lr 

        if sigmoid_output:
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()             

        for epoch in range(0, ops.epochs):
            learning_rate = optimizer.param_groups[0]['lr']
            print('Start training epoch {}. Learning rate {}'.format(epoch, learning_rate))            
            training_loss = 0.0
            valid_loss = 0.0
            model_.train()

            # For other lr_scheduler check torch.optim.lr_scheduler 
            # https://pytorch.org/docs/stable/optim.html
            if best_loss > training_loss:
                flag_change_lr_cnt = 0
                best_loss = training_loss
            else:
                flag_change_lr_cnt += 1
                if flag_change_lr_cnt > 10:
                    init_lr = init_lr*ops.lr_decay
                    set_learning_rate(optimizer, init_lr)
                    flag_change_lr_cnt = 0

            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device) 
  
                output = model_(inputs.float())
  
                if sigmoid_output:
                    output = output.squeeze()
                    targets = targets.squeeze().float()
                else:
                    output = output.squeeze() # bs,dim,1,1 > bs,dim           
                #print('targets and output', targets.shape, output.shape)

                loss = loss_fn(output, targets)

                loss.backward()
                optimizer.step()

                training_loss += loss.data.item() * inputs.size(0)  # batch size
            training_loss /= len(train_loader.dataset)
            
            model_.eval()
            num_correct = 0 
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model_(inputs.float())
            
                targets = targets.to(device) 
                if sigmoid_output:
                    output = output.squeeze()
                    targets = targets.squeeze().float()                    
                else:
                    output = output.squeeze() # bs,dim,1,1 > bs,dim

                loss = loss_fn(output,targets) 
                valid_loss += loss.data.item() * inputs.size(0)

                if sigmoid_output:           
                    correct = torch.eq(torch.round(output), targets).view(-1)
                else:
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
                
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)

            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
            valid_loss, num_correct / num_examples))
            torch.save(model_, ops.model_exp + '/{}_ep-{}_loss-{:.4f}.pth'.format(ops.experiment_name,epoch,valid_loss))



    except Exception as e:
        print('Exception : ',e) 
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])
        print('Exception  line : ', e.__traceback__.tb_lineno)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Classifier')
    parser.add_argument('--model_exp', type=str, default = './saved_weights',
        help = 'model_exp')
    parser.add_argument('--experiment_name', type=str, default = 'A',
        help = 'experiment_name')        
    parser.add_argument('--num_classes', type=int , default = 2,
        help = 'num_classes, 1 or 2')
    parser.add_argument('--init_lr', type=float, default = 1e-3,
        help = 'init learning Rate') # ini
    parser.add_argument('--lr_decay', type=float, default = 0.1,
        help = 'learningRate_decay') # learning rate decay
    parser.add_argument('--momentum', type=float, default = 0.9,
        help = 'momentum') 
    parser.add_argument('--batch_size', type=int, default = 256,
        help = 'batch_size') 
    parser.add_argument('--epochs', type=int, default = 1000,
        help = 'epochs') 

    args = parser.parse_args()

    trainer(ops = args)
