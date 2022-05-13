# binary-classifier-pytorch
PyTorch binary classifier with ResNet and custom dataset

# Description:

The code uses the pertained weights of ResNet18, replaces the last fc layer with output size 1 or 2 for my binary classifier. 
- If output size of 1 is used, sigmoid function is used on the output to give a value between 0 and 1, it’s then rounded to 0 or 1 for the class labels.
    - BCELoss() is used
- If output size of 2 is used, the index of the max value in the output tensor is used as the class label
    - CrossEntropyLoss() is used


# Conclusions and findings

- With learning rate = 1e-3, the loss and accuracy fluctuate and do not converge. Reducing learning rate to 1e-5 solves the problem

- Freezing all layers except the last fc layer result in very slow convergence and lower accuracy in the end

- CrossEntropyLoss() results in slightly better accuracy than BCELoss() for my dataset

